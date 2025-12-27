"""
Bulletproof External API Connection Manager
Ensures system never crashes due to external API failures

Key Features:
- Exponential backoff retry logic
- Circuit breaker pattern
- Timeout protection
- Rate limiting
- Health monitoring
- Automatic fallback strategies
"""

import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

# Global cache for CoinGecko sentiment data (5-minute TTL)
_sentiment_cache: Dict[str, Dict[str, Any]] = {}
_sentiment_cache_expiry: Dict[str, datetime] = {}

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests (API is down)
    HALF_OPEN = "half_open"  # Testing if API recovered


@dataclass
class APIStats:
    """Track API health statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_retries: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    avg_response_time: float = 0.0
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def failure_rate(self) -> float:
        """Calculate failure rate"""
        return 1.0 - self.success_rate()


class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures
    
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, block all requests
    - HALF_OPEN: Test if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker: Service recovered, closing circuit")
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit breaker: Opening circuit after {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN
    
    def can_request(self) -> bool:
        """Check if request is allowed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    logger.info("Circuit breaker: Attempting recovery (half-open)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
            return False
        
        # HALF_OPEN: Allow limited requests to test recovery
        return True


class BulletproofAPIClient:
    """
    Bulletproof API client with retry logic, circuit breaker, and health monitoring
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        timeout: int = 30,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        rate_limit_calls: int = 100,
        rate_limit_period: float = 60.0
    ):
        self.name = name
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        # Components
        self.circuit_breaker = CircuitBreaker()
        self.stats = APIStats()
        
        # Rate limiting
        self.rate_limit_calls = rate_limit_calls
        self.rate_limit_period = rate_limit_period
        self._rate_limit_timestamps: deque = deque()
        self._rate_limit_lock = asyncio.Lock()
    
    async def _wait_for_rate_limit(self):
        """Wait if rate limit exceeded"""
        async with self._rate_limit_lock:
            now = asyncio.get_event_loop().time()
            cutoff = now - self.rate_limit_period
            
            # Remove old timestamps
            while self._rate_limit_timestamps and self._rate_limit_timestamps[0] <= cutoff:
                self._rate_limit_timestamps.popleft()
            
            # Check if we hit the limit
            if len(self._rate_limit_timestamps) >= self.rate_limit_calls:
                sleep_time = self._rate_limit_timestamps[0] + self.rate_limit_period - now
                if sleep_time > 0:
                    logger.debug(f"{self.name}: Rate limit reached, sleeping {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
                    # Recursive call after sleep
                    await self._wait_for_rate_limit()
                    return
            
            # Add current timestamp
            self._rate_limit_timestamps.append(now)
    
    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        fallback: Optional[T] = None
    ) -> Optional[T]:
        """
        GET request with full bulletproofing
        
        Args:
            endpoint: API endpoint (will be appended to base_url)
            params: Query parameters
            fallback: Value to return if all retries fail
            
        Returns:
            Response data or fallback value
        """
        # Check circuit breaker
        if not self.circuit_breaker.can_request():
            logger.warning(f"{self.name}: Circuit breaker is OPEN, using fallback")
            self.stats.failed_requests += 1
            return fallback
        
        # Wait for rate limit
        await self._wait_for_rate_limit()
        
        # Track request
        self.stats.total_requests += 1
        request_start = datetime.now()
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}{endpoint}"
                
                # Increase connection pool to prevent saturation
                connector = aiohttp.TCPConnector(
                    limit=30,  # Increased from default 10
                    limit_per_host=20  # Increased from default 5
                )
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                    async with session.get(url, params=params) as response:
                        # Calculate response time
                        elapsed = (datetime.now() - request_start).total_seconds()
                        
                        # Update average response time
                        if self.stats.avg_response_time == 0:
                            self.stats.avg_response_time = elapsed
                        else:
                            self.stats.avg_response_time = (
                                self.stats.avg_response_time * 0.9 + elapsed * 0.1
                            )
                        
                        # Success
                        if response.status == 200:
                            data = await response.json()
                            
                            # Record success
                            self.stats.successful_requests += 1
                            self.stats.last_success = datetime.now()
                            self.circuit_breaker.record_success()
                            
                            logger.debug(
                                f"{self.name}: GET {endpoint} SUCCESS "
                                f"(attempt {attempt + 1}/{self.max_retries}, "
                                f"{elapsed:.2f}s)"
                            )
                            
                            return data
                        
                        # Rate limit error (429)
                        elif response.status == 429:
                            retry_after = response.headers.get('Retry-After', '60')
                            try:
                                wait_time = int(retry_after)
                            except:
                                wait_time = 60
                            
                            # Cap wait time at 10s to prevent startup hangs
                            wait_time = min(wait_time, 10)
                            
                            logger.warning(
                                f"{self.name}: Rate limited (429), waiting {wait_time}s (capped)"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        
                        # Client error (4xx) - don't retry
                        elif 400 <= response.status < 500:
                            error_text = await response.text()
                            logger.warning(
                                f"{self.name}: Client error {response.status} - {error_text[:200]}"
                            )
                            self.stats.failed_requests += 1
                            self.stats.recent_errors.append({
                                'timestamp': datetime.now(),
                                'status': response.status,
                                'error': error_text[:100]
                            })
                            return fallback
                        
                        # Server error (5xx) - retry
                        else:
                            error_text = await response.text()
                            logger.warning(
                                f"{self.name}: Server error {response.status}, "
                                f"will retry (attempt {attempt + 1}/{self.max_retries})"
                            )
                            
                            self.stats.recent_errors.append({
                                'timestamp': datetime.now(),
                                'status': response.status,
                                'error': error_text[:100]
                            })
                            
                            # Exponential backoff before retry
                            if attempt < self.max_retries - 1:
                                backoff = self.backoff_factor ** attempt
                                logger.debug(f"{self.name}: Backing off for {backoff}s")
                                await asyncio.sleep(backoff)
            
            except asyncio.TimeoutError:
                logger.warning(
                    f"{self.name}: Timeout after {self.timeout}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                
                self.stats.recent_errors.append({
                    'timestamp': datetime.now(),
                    'error': 'Timeout'
                })
                
                if attempt < self.max_retries - 1:
                    backoff = self.backoff_factor ** attempt
                    await asyncio.sleep(backoff)
            
            except aiohttp.ClientError as e:
                logger.warning(
                    f"{self.name}: Network error - {str(e)} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                
                self.stats.recent_errors.append({
                    'timestamp': datetime.now(),
                    'error': str(e)[:100]
                })
                
                if attempt < self.max_retries - 1:
                    backoff = self.backoff_factor ** attempt
                    await asyncio.sleep(backoff)
            
            except Exception as e:
                logger.error(
                    f"{self.name}: Unexpected error - {str(e)} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                
                self.stats.recent_errors.append({
                    'timestamp': datetime.now(),
                    'error': str(e)[:100]
                })
                
                if attempt < self.max_retries - 1:
                    backoff = self.backoff_factor ** attempt
                    await asyncio.sleep(backoff)
        
        # All retries failed
        logger.error(
            f"{self.name}: All {self.max_retries} retries failed for GET {endpoint}"
        )
        
        self.stats.failed_requests += 1
        self.stats.last_failure = datetime.now()
        self.stats.total_retries += self.max_retries - 1
        self.circuit_breaker.record_failure()
        
        return fallback
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get API health statistics"""
        return {
            'name': self.name,
            'circuit_state': self.circuit_breaker.state.value,
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'success_rate': self.stats.success_rate(),
            'failure_rate': self.stats.failure_rate(),
            'avg_response_time': f"{self.stats.avg_response_time:.2f}s",
            'last_success': self.stats.last_success.isoformat() if self.stats.last_success else None,
            'last_failure': self.stats.last_failure.isoformat() if self.stats.last_failure else None,
            'recent_errors_count': len(self.stats.recent_errors),
            'health': 'healthy' if self.stats.success_rate() > 0.95 else 'degraded' if self.stats.success_rate() > 0.80 else 'unhealthy'
        }


# Global API clients (singleton pattern)
_binance_client: Optional[BulletproofAPIClient] = None
_binance_futures_client: Optional[BulletproofAPIClient] = None
_coingecko_client: Optional[BulletproofAPIClient] = None


def get_binance_client() -> BulletproofAPIClient:
    """Get bulletproof Binance FUTURES API client (singleton) - USDC/USDT perpetual contracts"""
    global _binance_futures_client
    
    if _binance_futures_client is None:
        _binance_futures_client = BulletproofAPIClient(
            name="Binance Futures",
            base_url="https://fapi.binance.com",  # FUTURES API (supports both USDT and USDC)
            timeout=30,
            max_retries=3,
            backoff_factor=2.0,
            rate_limit_calls=90,  # Binance allows 100 req/min, use 90 to be safe
            rate_limit_period=60.0
        )
        logger.info("Binance FUTURES bulletproof API client initialized")
    
    return _binance_futures_client


def get_coingecko_client() -> BulletproofAPIClient:
    """Get bulletproof CoinGecko API client (singleton)"""
    global _coingecko_client
    
    if _coingecko_client is None:
        _coingecko_client = BulletproofAPIClient(
            name="CoinGecko",
            base_url="https://api.coingecko.com",
            timeout=30,
            max_retries=3,
            backoff_factor=2.0,
            rate_limit_calls=45,  # CoinGecko free tier: 50 req/min, use 45 to be safe
            rate_limit_period=60.0
        )
        logger.info("CoinGecko bulletproof API client initialized")
    
    return _coingecko_client


def get_all_api_health() -> Dict[str, Any]:
    """Get health status of all external APIs"""
    logger.info("[API_HEALTH] Aggregating external API health status")
    health: Dict[str, Any] = {}

    # Ensure clients are initialized so they appear in the health payload
    try:
        if _coingecko_client is None:
            get_coingecko_client()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning("[API_HEALTH] Unable to initialize CoinGecko client: %s", exc)

    # Futures client is the primary path used in the app (binance_ohlcv)
    if _binance_futures_client:
        health["binance_futures"] = _binance_futures_client.get_health_status()

    # Legacy spot client kept for completeness if instantiated elsewhere
    if _binance_client:
        health["binance_spot"] = _binance_client.get_health_status()

    if _coingecko_client:
        health["coingecko"] = _coingecko_client.get_health_status()

    # Overall system health
    if health:
        rates = [entry.get("success_rate", 0.0) for entry in health.values()]
        avg_success_rate = sum(rates) / len(rates) if rates else 0.0

        health["overall"] = {
            "status": "healthy" if avg_success_rate > 0.95 else "degraded" if avg_success_rate > 0.80 else "critical",
            "avg_success_rate": avg_success_rate,
            "apis_count": len(health),
            "timestamp": datetime.now().isoformat(),
        }

    return health


# Global health monitor singleton
_health_monitor = None


class APIHealthMonitor:
    """Global health monitor for all API services"""
    
    def __init__(self):
        self.services: Dict[str, APIStats] = {}
    
    def record_request(self, service: str, success: bool, response_time: float = 0.0):
        """Record an API request"""
        if service not in self.services:
            self.services[service] = APIStats()
        
        stats = self.services[service]
        stats.total_requests += 1
        
        if success:
            stats.successful_requests += 1
            stats.last_success = datetime.now()
        else:
            stats.failed_requests += 1
            stats.last_failure = datetime.now()
        
        # Update average response time
        if stats.total_requests == 1:
            stats.avg_response_time = response_time
        else:
            stats.avg_response_time = (
                stats.avg_response_time * (stats.total_requests - 1) + response_time
            ) / stats.total_requests
    
    def get_health_status(self, service: str) -> Dict[str, Any]:
        """Get health status for a service"""
        if service not in self.services:
            return {
                'service': service,
                'health': 'unknown',
                'total_requests': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0
            }
        
        stats = self.services[service]
        success_rate = stats.success_rate()
        
        return {
            'service': service,
            'health': 'healthy' if success_rate > 0.95 else 'degraded' if success_rate > 0.80 else 'critical',
            'total_requests': stats.total_requests,
            'successful_requests': stats.successful_requests,
            'failed_requests': stats.failed_requests,
            'success_rate': success_rate,
            'avg_response_time': stats.avg_response_time,
            'last_success': stats.last_success.isoformat() if stats.last_success else None,
            'last_failure': stats.last_failure.isoformat() if stats.last_failure else None
        }


def get_api_health_monitor() -> APIHealthMonitor:
    """Get global health monitor singleton"""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = APIHealthMonitor()
    
    return _health_monitor


def get_cached_sentiment(symbol: str, fetch_func: Callable) -> Optional[Dict[str, Any]]:
    """
    Get sentiment data with caching to reduce API calls
    
    Args:
        symbol: Trading symbol (e.g., 'BTC')
        fetch_func: Function to fetch fresh data if cache miss
    
    Returns:
        Cached or fresh sentiment data
    """
    global _sentiment_cache, _sentiment_cache_expiry
    
    now = datetime.now()
    
    # Check cache
    if symbol in _sentiment_cache:
        if symbol in _sentiment_cache_expiry and _sentiment_cache_expiry[symbol] > now:
            logger.debug(f"Sentiment cache HIT for {symbol}")
            return _sentiment_cache[symbol]
    
    # Cache miss - fetch fresh data
    logger.debug(f"Sentiment cache MISS for {symbol}, fetching...")
    try:
        result = fetch_func(symbol)
        
        # Store in cache with 5-minute expiry
        _sentiment_cache[symbol] = result
        _sentiment_cache_expiry[symbol] = now + timedelta(minutes=5)
        
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch sentiment for {symbol}: {e}")
        # Return stale cache if available
        return _sentiment_cache.get(symbol)


def create_bulletproof_client(
    base_url: str,
    service_name: str = "api",
    max_retries: int = 3,
    timeout: float = 30.0,
    rate_limit_calls: int = 100,
    rate_limit_period: float = 60.0
) -> BulletproofAPIClient:
    """Create a bulletproof API client with custom configuration"""
    return BulletproofAPIClient(
        name=service_name,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=2.0,
        rate_limit_calls=rate_limit_calls,
        rate_limit_period=rate_limit_period
    )
