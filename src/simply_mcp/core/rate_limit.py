"""Rate limiting system for HTTP transport.

This module provides token bucket-based rate limiting for API endpoints,
with per-key tracking and configurable limits.

Features:
- Token bucket algorithm implementation
- Per-API-key rate limiting
- Configurable request limits and time windows
- In-memory state tracking
- Automatic token replenishment
- Remaining quota info for responses

Example:
    >>> from simply_mcp.core.rate_limit import RateLimiter, RateLimitConfig
    >>>
    >>> # Create rate limiter
    >>> limiter = RateLimiter()
    >>>
    >>> # Add key with limits
    >>> config = RateLimitConfig(max_requests=100, window_seconds=3600)
    >>> limiter.add_key("api_key_1", config)
    >>>
    >>> # Check if request is allowed
    >>> allowed, info = limiter.check_limit("api_key_1")
    >>> if allowed:
    ...     print(f"Request allowed, {info.remaining} remaining")
"""

import time
from dataclasses import dataclass

from simply_mcp.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        max_requests: Maximum number of requests allowed
        window_seconds: Time window in seconds for the rate limit
    """

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_requests <= 0:
            raise ValueError("max_requests must be positive")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")


@dataclass
class RateLimitInfo:
    """Information about rate limit status.

    Attributes:
        allowed: Whether the request is allowed
        remaining: Number of requests remaining in current window
        limit: Maximum requests allowed in window
        reset_at: Unix timestamp when the limit resets
        retry_after: Seconds to wait before retrying (if not allowed)
    """

    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: float | None = None


class TokenBucket:
    """Token bucket implementation for rate limiting.

    This class implements the token bucket algorithm where:
    - Tokens are added at a constant rate
    - Each request consumes one token
    - Requests are allowed only if tokens are available

    The bucket has a maximum capacity (max_requests) and refills
    at a rate of (max_requests / window_seconds) tokens per second.

    Example:
        >>> bucket = TokenBucket(max_requests=100, window_seconds=3600)
        >>> if bucket.consume():
        ...     print("Request allowed")
        ... else:
        ...     print("Rate limit exceeded")
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        """Initialize token bucket.

        Args:
            max_requests: Maximum number of tokens (capacity)
            window_seconds: Time window for refill rate calculation
        """
        self.capacity = max_requests
        self.window_seconds = window_seconds

        # Calculate refill rate (tokens per second)
        self.refill_rate = max_requests / window_seconds

        # Current token count (start with full bucket)
        self.tokens = float(max_requests)

        # Last refill timestamp
        self.last_refill = time.time()

        logger.debug(
            f"Token bucket created: capacity={max_requests}, window={window_seconds}s",
            extra={
                "capacity": max_requests,
                "window_seconds": window_seconds,
                "refill_rate": self.refill_rate,
            }
        )

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Calculate tokens to add based on elapsed time
        tokens_to_add = elapsed * self.refill_rate

        # Update token count (cap at capacity)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

        logger.debug(
            f"Tokens refilled: {tokens_to_add:.2f} added, {self.tokens:.2f} current",
            extra={
                "tokens_added": tokens_to_add,
                "current_tokens": self.tokens,
                "elapsed_seconds": elapsed,
            }
        )

    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume (default: 1)

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        # Refill tokens based on time elapsed
        self._refill()

        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            logger.debug(
                f"Tokens consumed: {tokens}, {self.tokens:.2f} remaining",
                extra={
                    "tokens_consumed": tokens,
                    "remaining_tokens": self.tokens,
                }
            )
            return True

        logger.debug(
            f"Insufficient tokens: need {tokens}, have {self.tokens:.2f}",
            extra={
                "tokens_needed": tokens,
                "available_tokens": self.tokens,
            }
        )
        return False

    def get_remaining(self) -> int:
        """Get number of tokens currently available.

        Returns:
            Number of tokens available (rounded down)
        """
        self._refill()
        return int(self.tokens)

    def get_reset_time(self) -> float:
        """Get timestamp when bucket will be full again.

        Returns:
            Unix timestamp when bucket reaches capacity
        """
        self._refill()

        # If bucket is full, reset time is now
        if self.tokens >= self.capacity:
            return time.time()

        # Calculate time needed to fill bucket
        tokens_needed = self.capacity - self.tokens
        seconds_to_full = tokens_needed / self.refill_rate

        return time.time() + seconds_to_full

    def get_retry_after(self) -> float:
        """Get seconds to wait before one token is available.

        Returns:
            Seconds to wait until at least one token is available
        """
        self._refill()

        # If we have tokens, no need to wait
        if self.tokens >= 1:
            return 0.0

        # Calculate time needed for one token
        tokens_needed = 1.0 - self.tokens
        return tokens_needed / self.refill_rate


class RateLimiter:
    """Rate limiter with per-key tracking.

    This class manages rate limiting for multiple API keys, each with
    their own token bucket. It provides methods to check limits and
    get remaining quota information.

    Example:
        >>> limiter = RateLimiter()
        >>>
        >>> # Configure rate limit for a key
        >>> config = RateLimitConfig(max_requests=100, window_seconds=3600)
        >>> limiter.add_key("api_key_1", config)
        >>>
        >>> # Check if request is allowed
        >>> allowed, info = limiter.check_limit("api_key_1")
        >>> if allowed:
        ...     print(f"{info.remaining} requests remaining")
        ... else:
        ...     print(f"Rate limited, retry after {info.retry_after}s")
    """

    def __init__(self) -> None:
        """Initialize rate limiter with empty key tracking."""
        self._buckets: dict[str, TokenBucket] = {}
        self._configs: dict[str, RateLimitConfig] = {}
        logger.info("Rate limiter initialized")

    def add_key(self, key: str, config: RateLimitConfig) -> None:
        """Add or update rate limit configuration for a key.

        Args:
            key: API key or identifier
            config: Rate limit configuration
        """
        bucket = TokenBucket(
            max_requests=config.max_requests,
            window_seconds=config.window_seconds,
        )
        self._buckets[key] = bucket
        self._configs[key] = config

        logger.info(
            "Rate limit configured for key",
            extra={
                "key_prefix": key[:8] + "..." if len(key) > 8 else key,
                "max_requests": config.max_requests,
                "window_seconds": config.window_seconds,
            }
        )

    def remove_key(self, key: str) -> bool:
        """Remove rate limit tracking for a key.

        Args:
            key: API key or identifier

        Returns:
            True if key was removed, False if key didn't exist
        """
        if key in self._buckets:
            del self._buckets[key]
            del self._configs[key]
            logger.info(
                "Rate limit removed for key",
                extra={"key_prefix": key[:8] + "..." if len(key) > 8 else key}
            )
            return True

        logger.warning("Attempted to remove non-existent rate limit key")
        return False

    def check_limit(self, key: str) -> tuple[bool, RateLimitInfo]:
        """Check if a request is allowed under the rate limit.

        This method checks the rate limit for a key and returns
        detailed information about the limit status.

        Args:
            key: API key or identifier

        Returns:
            Tuple of (allowed, info)
            - allowed: True if request is allowed
            - info: RateLimitInfo with detailed status
        """
        # Check if key has rate limiting configured
        if key not in self._buckets:
            logger.warning(
                "Rate limit check for unconfigured key",
                extra={"key_prefix": key[:8] + "..." if len(key) > 8 else key}
            )
            # If no rate limit configured, allow by default
            # This makes rate limiting opt-in per key
            return True, RateLimitInfo(
                allowed=True,
                remaining=999999,
                limit=999999,
                reset_at=time.time() + 3600,
            )

        bucket = self._buckets[key]
        config = self._configs[key]

        # Try to consume a token
        allowed = bucket.consume(1)

        # Build rate limit info
        info = RateLimitInfo(
            allowed=allowed,
            remaining=bucket.get_remaining(),
            limit=config.max_requests,
            reset_at=bucket.get_reset_time(),
            retry_after=bucket.get_retry_after() if not allowed else None,
        )

        if allowed:
            logger.debug(
                "Rate limit check: allowed",
                extra={
                    "key_prefix": key[:8] + "..." if len(key) > 8 else key,
                    "remaining": info.remaining,
                    "limit": info.limit,
                }
            )
        else:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "key_prefix": key[:8] + "..." if len(key) > 8 else key,
                    "limit": info.limit,
                    "retry_after": info.retry_after,
                }
            )

        return allowed, info

    def get_status(self, key: str) -> RateLimitInfo | None:
        """Get current rate limit status for a key without consuming tokens.

        Args:
            key: API key or identifier

        Returns:
            RateLimitInfo if key is configured, None otherwise
        """
        if key not in self._buckets:
            return None

        bucket = self._buckets[key]
        config = self._configs[key]

        return RateLimitInfo(
            allowed=bucket.get_remaining() > 0,
            remaining=bucket.get_remaining(),
            limit=config.max_requests,
            reset_at=bucket.get_reset_time(),
        )

    def reset_key(self, key: str) -> bool:
        """Reset rate limit for a key (refill bucket to capacity).

        Args:
            key: API key or identifier

        Returns:
            True if key was reset, False if key doesn't exist
        """
        if key not in self._buckets:
            logger.warning("Attempted to reset non-existent rate limit key")
            return False

        # Get current config
        config = self._configs[key]

        # Create new bucket with full tokens
        self._buckets[key] = TokenBucket(
            max_requests=config.max_requests,
            window_seconds=config.window_seconds,
        )

        logger.info(
            "Rate limit reset for key",
            extra={"key_prefix": key[:8] + "..." if len(key) > 8 else key}
        )

        return True

    def get_all_keys(self) -> list[str]:
        """Get list of all keys with rate limiting configured.

        Returns:
            List of API keys
        """
        return list(self._buckets.keys())

    def count_keys(self) -> int:
        """Get count of keys with rate limiting configured.

        Returns:
            Number of keys
        """
        return len(self._buckets)


__all__ = [
    "RateLimitConfig",
    "RateLimitInfo",
    "TokenBucket",
    "RateLimiter",
]
