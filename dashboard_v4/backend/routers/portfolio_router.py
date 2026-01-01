from fastapi import APIRouter
from schemas import Portfolio
import random
import logging
from services.quantum_client import quantum_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolio", tags=["Portfolio"])

@router.get("/status", response_model=Portfolio)
async def get_portfolio_status():
    """Get portfolio status with PnL and exposure metrics
    
    TESTNET MODE: Fetches REAL positions directly from Binance TESTNET Futures API.
    Shows actual account balance, positions, and PnL.
    """
    import os
    
    # Try to get REAL data from Binance TESTNET API
    try:
        from binance.client import Client
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        use_testnet = os.getenv('TESTNET', 'true').lower() == 'true'
        
        if not api_key or not api_secret:
            raise Exception("Binance API keys not configured")
        
        # Connect to TESTNET Futures (python-binance requires testnet URL for futures)
        client = Client(api_key, api_secret, testnet=False)
        
        # Override base endpoint manually for TESTNET futures
        if use_testnet:
            client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            client.FUTURES_URL_V2 = "https://testnet.binancefuture.com/fapi/v2"
        
        # Get account info with positions
        account = client.futures_account()
        
        # Parse positions
        positions = account.get('positions', [])
        active_positions = []
        total_pnl = 0.0
        total_position_value = 0.0
        
        for pos in positions:
            amt = float(pos.get('positionAmt', 0))
            if amt != 0:
                unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                notional = float(pos.get('notional', 0))
                
                active_positions.append({
                    'symbol': pos.get('symbol'),
                    'size': amt,
                    'entry': float(pos.get('entryPrice', 0)),
                    'pnl': unrealized_pnl
                })
                
                total_pnl += unrealized_pnl
                total_position_value += abs(notional)
        
        # Get balance
        total_balance = float(account.get('totalWalletBalance', 0))
        
        # Calculate exposure (position value / balance)
        exposure = min(total_position_value / total_balance, 1.0) if total_balance > 0 else 0.0
        
        # Calculate drawdown (simplified - from unrealized losses)
        drawdown_pct = abs(total_pnl / total_balance) if total_pnl < 0 and total_balance > 0 else 0.0
        
        mode = "🧪 TESTNET" if use_testnet else "🔴 MAINNET"
        logger.info(f"✅ {mode} Portfolio: {len(active_positions)} positions, Balance: ${total_balance:.2f}, PnL: ${total_pnl:.2f}, Exposure: {exposure:.1%}")
        
        return Portfolio(
            pnl=round(total_pnl, 2),
            exposure=round(exposure, 2),
            drawdown=round(drawdown_pct, 3),
            positions=len(active_positions)
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch Binance account data: {e}")
    
    # Fallback: Return empty state
    logger.warning("⚠️ Using fallback - no Binance API access")
    return Portfolio(
        pnl=0.0,
        exposure=0.0,
        drawdown=0.0,
        positions=0
    )
