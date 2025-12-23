"""
Funding Rate Filter - Prevents trading symbols with excessive funding costs

High funding rates can erode profits on perpetual futures positions.
This filter blocks trades when funding costs exceed acceptable thresholds.
"""
import logging
from typing import Dict, Optional
from binance.client import Client
import os

logger = logging.getLogger(__name__)


class FundingRateFilter:
    """
    Filters out symbols with excessive funding rates to prevent
    slow erosion of capital through funding fees.
    
    Funding Rate Examples:
    - 0.01% (0.0001): Normal, acceptable
    - 0.05% (0.0005): Moderate, caution
    - 0.10% (0.001): High, avoid
    - 0.30% (0.003): Extreme, definitely avoid
    
    Daily cost = notional × funding_rate × 3 (3 times per day)
    Monthly cost = daily_cost × 30
    """
    
    def __init__(
        self,
        max_funding_rate: float = 0.001,  # 0.1% per 8 hours = maximum acceptable
        warn_funding_rate: float = 0.0005,  # 0.05% per 8 hours = warning threshold
        use_testnet: bool = None
    ):
        self.max_funding_rate = max_funding_rate
        self.warn_funding_rate = warn_funding_rate
        
        # Binance client setup
        if use_testnet is None:
            use_testnet = os.getenv("USE_BINANCE_TESTNET", "false").lower() == "true"
        
        if use_testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            api_secret = os.getenv("BINANCE_TESTNET_SECRET_KEY")
            self.client = Client(api_key, api_secret, testnet=True)
            self.client.API_URL = 'https://testnet.binancefuture.com'
            logger.info("[TEST_TUBE] Funding Rate Filter: Using testnet API")
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            api_secret = os.getenv("BINANCE_API_SECRET")
            self.client = Client(api_key, api_secret)
            logger.info("[MONEY_BAG] Funding Rate Filter: Using live API")
        
        # Cache for funding rates (1 minute TTL)
        self.funding_cache: Dict[str, tuple[float, float]] = {}  # symbol -> (rate, timestamp)
        self.cache_ttl = 60  # seconds
        
        logger.info(
            f"[MONEY_WITH_WINGS] Funding Rate Filter initialized: "
            f"Max={self.max_funding_rate*100:.3f}%, Warn={self.warn_funding_rate*100:.3f}%"
        )
    
    def get_funding_features(self, symbol: str) -> Optional[Dict]:
        """
        Get funding rate features for AI Engine.
        Returns: Dict with current_funding, funding_delta, crowd_bias
        """
        try:
            current_funding = self.get_current_funding_rate(symbol)
            if current_funding is None:
                return None
            
            # Funding delta (change from historical) - placeholder for now
            funding_delta = 0.0
            
            # Crowd bias: positive funding = crowd is long, negative = crowd is short
            crowd_bias = current_funding * 100  # Scale for readability
            
            return {
                "current_funding": current_funding,
                "funding_delta": funding_delta,
                "crowd_bias": crowd_bias
            }
        except Exception as e:
            logger.error(f"Error getting funding features for {symbol}: {e}")
            return None
    
    def should_block_trade(self, symbol: str) -> Dict:
        """
        Check if trade should be blocked due to excessive funding.
        Returns: Dict with blocked (bool), reason (str), funding_rate (float)
        """
        try:
            funding_rate = self.get_current_funding_rate(symbol)
            if funding_rate is None:
                return {"blocked": False, "reason": "No funding data available"}
            
            abs_funding = abs(funding_rate)
            
            if abs_funding >= self.max_funding_rate:
                return {
                    "blocked": True,
                    "reason": f"Funding rate too high: {abs_funding:.5f}",
                    "funding_rate": abs_funding
                }
            elif abs_funding >= self.warn_funding_rate:
                logger.warning(
                    f"[FUNDING WARNING] {symbol} funding rate: {abs_funding:.5f} "
                    f"(threshold: {self.warn_funding_rate:.5f})"
                )
            
            return {"blocked": False, "funding_rate": abs_funding}
        except Exception as e:
            logger.error(f"Error checking funding for {symbol}: {e}")
            return {"blocked": False, "reason": f"Error: {e}"}
    
    def get_current_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Get current funding rate for a symbol.
        
        Returns:
            Funding rate as decimal (e.g., 0.0001 = 0.01%)
            None if unavailable
        """
        try:
            # Check cache first
            import time
            now = time.time()
            if symbol in self.funding_cache:
                rate, timestamp = self.funding_cache[symbol]
                if now - timestamp < self.cache_ttl:
                    return rate
            
            # Fetch from API
            funding_rate = self.client.futures_funding_rate(symbol=symbol, limit=1)
            
            if funding_rate and len(funding_rate) > 0:
                rate = float(funding_rate[0]['fundingRate'])
                self.funding_cache[symbol] = (rate, now)
                return rate
            
            return None
            
        except Exception as e:
            logger.warning(f"Could not fetch funding rate for {symbol}: {e}")
            return None
    
    def calculate_funding_cost(
        self,
        symbol: str,
        position_size_usdt: float,
        funding_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate funding cost estimates.
        
        Args:
            symbol: Trading symbol
            position_size_usdt: Position size in USDT (notional value)
            funding_rate: Override rate (if None, fetches current)
        
        Returns:
            Dict with cost_per_8h, daily_cost, monthly_cost, annual_cost
        """
        if funding_rate is None:
            funding_rate = self.get_current_funding_rate(symbol)
        
        if funding_rate is None:
            return {
                "cost_per_8h": 0.0,
                "daily_cost": 0.0,
                "monthly_cost": 0.0,
                "annual_cost": 0.0
            }
        
        # Funding happens every 8 hours (3 times per day)
        cost_per_8h = abs(position_size_usdt * funding_rate)
        daily_cost = cost_per_8h * 3
        monthly_cost = daily_cost * 30
        annual_cost = daily_cost * 365
        
        return {
            "cost_per_8h": cost_per_8h,
            "daily_cost": daily_cost,
            "monthly_cost": monthly_cost,
            "annual_cost": annual_cost
        }
    
    def should_block_trade(
        self,
        symbol: str,
        position_size_usdt: float,
        is_long: bool
    ) -> tuple[bool, str]:
        """
        Determine if trade should be blocked due to high funding rate.
        
        Args:
            symbol: Trading symbol
            position_size_usdt: Intended position size in USDT
            is_long: True for LONG, False for SHORT
        
        Returns:
            (should_block, reason)
            - should_block: True if trade should be blocked
            - reason: Human-readable explanation
        """
        funding_rate = self.get_current_funding_rate(symbol)
        
        if funding_rate is None:
            # Cannot determine - allow trade but warn
            return False, "Funding rate unavailable - trade allowed"
        
        # Determine if funding is favorable or unfavorable
        # Positive funding rate = Longs pay Shorts
        # Negative funding rate = Shorts pay Longs
        
        paying_funding = (is_long and funding_rate > 0) or (not is_long and funding_rate < 0)
        
        if not paying_funding:
            # We're receiving funding, not paying - this is good!
            return False, f"Receiving funding: {abs(funding_rate)*100:.4f}% per 8h"
        
        # We're paying funding - check threshold
        abs_rate = abs(funding_rate)
        
        if abs_rate > self.max_funding_rate:
            # BLOCK: Funding rate too high
            costs = self.calculate_funding_cost(symbol, position_size_usdt, funding_rate)
            return True, (
                f"BLOCKED: Funding rate {abs_rate*100:.4f}% exceeds max {self.max_funding_rate*100:.4f}% "
                f"(Daily cost: ${costs['daily_cost']:.2f}, Monthly: ${costs['monthly_cost']:.2f})"
            )
        
        elif abs_rate > self.warn_funding_rate:
            # WARN: Funding rate is moderate
            costs = self.calculate_funding_cost(symbol, position_size_usdt, funding_rate)
            logger.warning(
                f"[WARNING] {symbol}: High funding rate {abs_rate*100:.4f}% "
                f"(Daily cost: ${costs['daily_cost']:.2f}, Monthly: ${costs['monthly_cost']:.2f})"
            )
            return False, f"Warning: High funding {abs_rate*100:.4f}%/8h"
        
        else:
            # OK: Funding rate acceptable
            return False, f"Funding OK: {abs_rate*100:.4f}%/8h"
    
    def get_funding_report(self, symbol: str, position_size_usdt: float, is_long: bool) -> str:
        """
        Generate detailed funding report for a symbol.
        
        Args:
            symbol: Trading symbol
            position_size_usdt: Position size in USDT
            is_long: True for LONG, False for SHORT
        
        Returns:
            Formatted report string
        """
        funding_rate = self.get_current_funding_rate(symbol)
        
        if funding_rate is None:
            return f"{symbol}: Funding rate unavailable"
        
        costs = self.calculate_funding_cost(symbol, position_size_usdt, funding_rate)
        paying = (is_long and funding_rate > 0) or (not is_long and funding_rate < 0)
        direction = "PAYING" if paying else "RECEIVING"
        
        report = f"""
{symbol} Funding Analysis:
  Rate: {funding_rate*100:.4f}% per 8h
  Direction: {direction} funding ({'LONG' if is_long else 'SHORT'} position)
  Position Size: ${position_size_usdt:.2f}
  
  Cost Estimates:
    Per 8h:   ${costs['cost_per_8h']:.2f}
    Daily:    ${costs['daily_cost']:.2f}
    Monthly:  ${costs['monthly_cost']:.2f}
    Annual:   ${costs['annual_cost']:.2f}
"""
        return report.strip()


# Singleton instance
_funding_filter_instance = None


def get_funding_filter() -> FundingRateFilter:
    """Get global funding filter instance."""
    global _funding_filter_instance
    if _funding_filter_instance is None:
        _funding_filter_instance = FundingRateFilter()
    return _funding_filter_instance
