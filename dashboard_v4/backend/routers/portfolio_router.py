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
    
    Fetches REAL data from Redis (quantum:portfolio:realtime key)
    Updated by binance_pnl_tracker service.
    """
    import json
    
    # Get data from Redis
    try:
        # Try to get from quantum_client service
        redis_data = quantum_client.get_portfolio_status()
        
        if redis_data and redis_data != b'null':
            # Parse Redis data
            if isinstance(redis_data, bytes):
                redis_data = redis_data.decode('utf-8')
            
            # Handle Python dict format from Redis
            import ast
            try:
                portfolio = ast.literal_eval(redis_data)
            except:
                portfolio = json.loads(redis_data)
            
            total_pnl = float(portfolio.get('total_unrealized_pnl', 0))
            total_equity = float(portfolio.get('total_equity', 1))
            total_margin = float(portfolio.get('total_margin', 0))
            num_positions = int(portfolio.get('num_positions', 0))
            
            # Calculate exposure (margin / equity)
            exposure = min(total_margin / total_equity, 1.0) if total_equity > 0 else 0.0
            
            # Calculate drawdown (simplified)
            drawdown_pct = abs(total_pnl / total_equity) if total_pnl < 0 and total_equity > 0 else 0.0
            
            logger.info(f"✅ Portfolio from Redis: {num_positions} positions, PnL: ${total_pnl:.2f}, Exposure: {exposure:.1%}")
            
            return Portfolio(
                pnl=round(total_pnl, 2),
                exposure=round(exposure, 2),
                drawdown=round(drawdown_pct, 3),
                positions=num_positions
            )
    except Exception as e:
        logger.error(f"❌ Failed to fetch from Redis: {e}")
    
    # Fallback: Return empty state
    logger.warning("⚠️ Using fallback - no Redis data available")
    return Portfolio(
        pnl=0.0,
        exposure=0.0,
        drawdown=0.0,
        positions=0
    )
