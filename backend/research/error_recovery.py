"""
Error recovery and resilience utilities for Strategy Generator AI.

Provides retry logic, circuit breakers, and error handling patterns.
"""

import logging
import time
import functools
from typing import Callable, Any, Optional, Type
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.
    
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check circuit state
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Success - reset circuit
            self._on_success()
            return result
            
        except self.expected_exception as e:
            # Failure - record and potentially open circuit
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )
            
            raise last_exception
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    default=None,
    error_message: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Execute function with error handling, returning default on failure.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value to return on error
        error_message: Custom error message
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error executing {func.__name__}"
        logger.error(f"{msg}: {e}", exc_info=True)
        return default


class RateLimiter:
    """
    Rate limiter for API calls.
    
    Prevents exceeding API rate limits by throttling requests.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Maximum calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time: Optional[datetime] = None
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limit."""
        if self.last_call_time is None:
            self.last_call_time = datetime.utcnow()
            return
        
        elapsed = (datetime.utcnow() - self.last_call_time).total_seconds()
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = datetime.utcnow()


class ErrorBudget:
    """
    Error budget tracker for SLO monitoring.
    
    Tracks error rates and enforces error budgets.
    """
    
    def __init__(self, budget_percent: float = 1.0, window_hours: int = 24):
        """
        Initialize error budget.
        
        Args:
            budget_percent: Allowed error percentage (e.g., 1.0 = 1%)
            window_hours: Time window for tracking
        """
        self.budget_percent = budget_percent
        self.window = timedelta(hours=window_hours)
        self.events: list[tuple[datetime, bool]] = []  # (timestamp, success)
    
    def record(self, success: bool):
        """Record an event outcome."""
        self.events.append((datetime.utcnow(), success))
        self._cleanup_old_events()
    
    def _cleanup_old_events(self):
        """Remove events outside the time window."""
        cutoff = datetime.utcnow() - self.window
        self.events = [(t, s) for t, s in self.events if t >= cutoff]
    
    def get_error_rate(self) -> float:
        """Calculate current error rate percentage."""
        if not self.events:
            return 0.0
        
        self._cleanup_old_events()
        errors = sum(1 for _, success in self.events if not success)
        return (errors / len(self.events)) * 100.0
    
    def is_budget_exhausted(self) -> bool:
        """Check if error budget is exhausted."""
        return self.get_error_rate() > self.budget_percent
    
    def remaining_budget(self) -> float:
        """Calculate remaining error budget percentage."""
        return max(0.0, self.budget_percent - self.get_error_rate())


# Global instances
binance_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=300,  # 5 minutes
    expected_exception=Exception
)

binance_rate_limiter = RateLimiter(calls_per_minute=50)

generation_error_budget = ErrorBudget(budget_percent=5.0, window_hours=24)
