"""
Error handling utilities for robust API and processing operations.
Includes retry logic, rate limiting, and graceful degradation.
"""

import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from loguru import logger
import asyncio


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_backoff(max_retries=3, initial_delay=1.0)
        def fetch_data():
            return requests.get(url)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
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
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}"
                        )
            
            raise RetryExhaustedError(
                f"Failed after {max_retries + 1} attempts: {last_exception}"
            )
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """
    Async version of retry_with_backoff decorator.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated async function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Async attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            f"All {max_retries + 1} async attempts failed for {func.__name__}"
                        )
            
            raise RetryExhaustedError(
                f"Async failed after {max_retries + 1} attempts: {last_exception}"
            )
        
        return wrapper
    return decorator


class RateLimiter:
    """
    Rate limiter for API calls using token bucket algorithm.
    Ensures we don't exceed API rate limits.
    """
    
    def __init__(self, calls_per_minute: int = 60):
        """
        Initialize RateLimiter.
        
        Args:
            calls_per_minute: Maximum number of calls allowed per minute
        """
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute  # Seconds between calls
        self.last_call_time = 0.0
    
    def wait_if_needed(self) -> None:
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.interval:
            sleep_time = self.interval - time_since_last_call
            logger.debug(f"Rate limit: sleeping for {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    async def async_wait_if_needed(self) -> None:
        """Async version of wait_if_needed."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.interval:
            sleep_time = self.interval - time_since_last_call
            logger.debug(f"Rate limit: async sleeping for {sleep_time:.3f}s")
            await asyncio.sleep(sleep_time)
        
        self.last_call_time = time.time()


class CircuitBreaker:
    """
    Circuit breaker pattern for handling failing services.
    Prevents cascading failures by stopping requests to failing services.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize CircuitBreaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is OPEN or function fails
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            logger.info("Circuit breaker recovered, state: CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(
                f"Circuit breaker OPEN after {self.failure_count} failures"
            )


def handle_errors_gracefully(default_return: Any = None) -> Callable:
    """
    Decorator for graceful error handling.
    Returns default value instead of raising exceptions.
    
    Args:
        default_return: Value to return on error
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return default_return
        
        return wrapper
    return decorator
