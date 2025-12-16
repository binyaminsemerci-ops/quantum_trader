"""
Performance optimization utilities for Strategy Generator AI.

Includes caching, parallel execution, and resource management.
"""

import logging
from typing import Any, Callable, Optional
from functools import lru_cache, wraps
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimedCache:
    """
    Time-based cache with automatic expiration.
    
    Unlike functools.lru_cache, entries expire after a timeout.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize timed cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self.ttl_seconds = ttl_seconds
        self.cache: dict[Any, tuple[Any, datetime]] = {}
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        
        # Check expiration
        if (datetime.utcnow() - timestamp).total_seconds() > self.ttl_seconds:
            del self.cache[key]
            return None
        
        return value
    
    def set(self, key: Any, value: Any):
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, datetime.utcnow())
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self.cache)


def timed_cache(ttl_seconds: int = 3600):
    """
    Decorator for caching function results with expiration.
    
    Args:
        ttl_seconds: Cache entry time-to-live
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        cache = TimedCache(ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = (args, tuple(sorted(kwargs.items())))
            
            # Check cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            return result
        
        # Expose cache management
        wrapper.cache_clear = cache.clear
        wrapper.cache_size = cache.size
        
        return wrapper
    
    return decorator


def parallel_map(
    func: Callable,
    items: list,
    max_workers: int = 4,
    use_processes: bool = False
) -> list:
    """
    Execute function on items in parallel.
    
    Args:
        func: Function to execute
        items: List of items to process
        max_workers: Maximum parallel workers
        use_processes: Use processes instead of threads
        
    Returns:
        List of results in same order as items
    """
    if not items:
        return []
    
    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(func, item): i for i, item in enumerate(items)}
        
        # Collect results in order
        results = [None] * len(items)
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as e:
                logger.error(f"Parallel task {index} failed: {e}")
                results[index] = None
    
    return results


class BatchProcessor:
    """
    Process items in batches with rate limiting.
    
    Useful for API calls with rate limits.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        delay_between_batches: float = 1.0
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items per batch
            delay_between_batches: Seconds to wait between batches
        """
        self.batch_size = batch_size
        self.delay = delay_between_batches
    
    def process(
        self,
        items: list,
        processor: Callable
    ) -> list:
        """
        Process items in batches.
        
        Args:
            items: Items to process
            processor: Function to call on each item
            
        Returns:
            List of results
        """
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            
            # Process batch
            batch_results = [processor(item) for item in batch]
            results.extend(batch_results)
            
            # Delay before next batch (except for last batch)
            if i + self.batch_size < len(items):
                time.sleep(self.delay)
        
        return results


class ResourceMonitor:
    """
    Monitor resource usage (memory, CPU).
    
    Can be used to implement backpressure or throttling.
    """
    
    def __init__(
        self,
        max_memory_mb: Optional[int] = None,
        check_interval: int = 60
    ):
        """
        Initialize resource monitor.
        
        Args:
            max_memory_mb: Maximum memory usage threshold in MB
            check_interval: Seconds between checks
        """
        self.max_memory_mb = max_memory_mb
        self.check_interval = check_interval
        self.last_check: Optional[datetime] = None
    
    def should_throttle(self) -> bool:
        """Check if resource limits are exceeded."""
        # Only check periodically
        if self.last_check:
            elapsed = (datetime.utcnow() - self.last_check).total_seconds()
            if elapsed < self.check_interval:
                return False
        
        self.last_check = datetime.utcnow()
        
        # Check memory usage
        if self.max_memory_mb:
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > self.max_memory_mb:
                    logger.warning(
                        f"⚠️  High memory usage: {memory_mb:.0f}MB "
                        f"(limit: {self.max_memory_mb}MB)"
                    )
                    return True
            except ImportError:
                pass  # psutil not available
        
        return False


class StrategyCache:
    """
    Specialized cache for strategy configurations and statistics.
    
    Optimized for strategy generator use cases.
    """
    
    def __init__(self, max_strategies: int = 1000):
        """
        Initialize strategy cache.
        
        Args:
            max_strategies: Maximum cached strategies
        """
        self.max_strategies = max_strategies
        self.strategies: dict[str, Any] = {}
        self.stats: dict[str, Any] = {}
        self.access_times: dict[str, datetime] = {}
    
    def put_strategy(self, strategy_id: str, config: Any):
        """Cache strategy configuration."""
        self._ensure_capacity()
        self.strategies[strategy_id] = config
        self.access_times[strategy_id] = datetime.utcnow()
    
    def get_strategy(self, strategy_id: str) -> Optional[Any]:
        """Get cached strategy configuration."""
        if strategy_id in self.strategies:
            self.access_times[strategy_id] = datetime.utcnow()
            return self.strategies[strategy_id]
        return None
    
    def put_stats(self, strategy_id: str, stats: Any):
        """Cache strategy statistics."""
        self.stats[strategy_id] = stats
    
    def get_stats(self, strategy_id: str) -> Optional[Any]:
        """Get cached strategy statistics."""
        return self.stats.get(strategy_id)
    
    def _ensure_capacity(self):
        """Evict least recently used strategies if at capacity."""
        if len(self.strategies) >= self.max_strategies:
            # Find least recently used
            lru_id = min(self.access_times, key=self.access_times.get)
            
            # Evict
            del self.strategies[lru_id]
            del self.access_times[lru_id]
            if lru_id in self.stats:
                del self.stats[lru_id]
            
            logger.debug(f"Evicted strategy {lru_id} from cache")
    
    def clear(self):
        """Clear all cached data."""
        self.strategies.clear()
        self.stats.clear()
        self.access_times.clear()


# Global cache instance
strategy_cache = StrategyCache(max_strategies=1000)
