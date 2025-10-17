"""Prometheus Metrics for HTTP Transport.

This module provides comprehensive metrics collection for HTTP transport:
- Request count and latency by endpoint and status code
- Authentication failures by reason
- Rate limit hits by key
- Tool execution performance
- Active connections and request queue size

Metrics are exposed in Prometheus format and can be scraped by monitoring systems.
"""

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type checking only - imports always succeed for static analysis
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
else:
    # Runtime behavior - handle optional prometheus gracefully
    try:
        from prometheus_client import (
            CONTENT_TYPE_LATEST,
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
            Info,
            generate_latest,
        )
        PROMETHEUS_AVAILABLE = True
    except ImportError:
        PROMETHEUS_AVAILABLE = False
        # Provide stub implementations for runtime
        Counter = Any
        Gauge = Any
        Histogram = Any
        Info = Any
        generate_latest = Any
        CONTENT_TYPE_LATEST = Any
        CollectorRegistry = Any

from simply_mcp.core.logger import get_logger

logger = get_logger(__name__)


class HttpMetrics:
    """Prometheus metrics for HTTP transport.

    This class manages all metrics for the HTTP transport layer including:
    - Request metrics (count, latency, size)
    - Authentication metrics (failures, successes)
    - Rate limiting metrics (hits, denials)
    - Tool execution metrics (count, latency, errors)
    - System metrics (active connections, queue size)

    Example:
        >>> metrics = HttpMetrics()
        >>> metrics.record_request("/tools/echo", "POST", 200, 0.05)
        >>> metrics.record_auth_failure("invalid_token")
        >>> metrics.record_rate_limit_hit("user_123")
    """

    def __init__(self, registry: Any | None = None) -> None:
        """Initialize metrics collectors.

        Args:
            registry: Optional Prometheus registry (uses default if None)

        Raises:
            ImportError: If prometheus_client is not installed
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError(
                "prometheus_client is required for metrics. "
                "Install with: pip install prometheus-client"
            )

        self.registry = registry or CollectorRegistry()

        # Request metrics
        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )

        self.request_size = Histogram(
            "http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
            registry=self.registry,
        )

        self.response_size = Histogram(
            "http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint"],
            buckets=(100, 1000, 10000, 100000, 1000000, 10000000),
            registry=self.registry,
        )

        # Authentication metrics
        self.auth_success_count = Counter(
            "http_auth_success_total",
            "Successful authentication attempts",
            ["key_name"],
            registry=self.registry,
        )

        self.auth_failure_count = Counter(
            "http_auth_failure_total",
            "Failed authentication attempts",
            ["reason"],
            registry=self.registry,
        )

        # Rate limiting metrics
        self.rate_limit_hit_count = Counter(
            "http_rate_limit_hit_total",
            "Rate limit hits (within limit)",
            ["key"],
            registry=self.registry,
        )

        self.rate_limit_exceeded_count = Counter(
            "http_rate_limit_exceeded_total",
            "Rate limit exceeded (denied)",
            ["key"],
            registry=self.registry,
        )

        self.rate_limit_remaining = Gauge(
            "http_rate_limit_remaining",
            "Remaining requests in rate limit window",
            ["key"],
            registry=self.registry,
        )

        # Tool execution metrics
        self.tool_execution_count = Counter(
            "http_tool_execution_total",
            "Tool execution count",
            ["tool", "status"],
            registry=self.registry,
        )

        self.tool_execution_duration = Histogram(
            "http_tool_execution_duration_seconds",
            "Tool execution latency",
            ["tool"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
            registry=self.registry,
        )

        self.tool_execution_errors = Counter(
            "http_tool_execution_errors_total",
            "Tool execution errors",
            ["tool", "error_type"],
            registry=self.registry,
        )

        # System metrics
        self.active_connections = Gauge(
            "http_active_connections",
            "Number of active HTTP connections",
            registry=self.registry,
        )

        self.request_queue_size = Gauge(
            "http_request_queue_size",
            "Number of requests in queue",
            registry=self.registry,
        )

        # Server info
        self.server_info = Info(
            "http_server",
            "HTTP server information",
            registry=self.registry,
        )

        logger.info("HTTP metrics initialized")

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """Record HTTP request metrics.

        Args:
            endpoint: Request endpoint path
            method: HTTP method
            status_code: Response status code
            duration: Request duration in seconds
            request_size: Request body size in bytes
            response_size: Response body size in bytes
        """
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code),
        ).inc()

        self.request_duration.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        if request_size > 0:
            self.request_size.labels(
                method=method,
                endpoint=endpoint,
            ).observe(request_size)

        if response_size > 0:
            self.response_size.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)

        logger.debug(
            f"Recorded request: {method} {endpoint} {status_code} {duration:.3f}s",
            extra={
                "context": {
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "duration": duration,
                }
            },
        )

    def record_auth_success(self, key_name: str) -> None:
        """Record successful authentication.

        Args:
            key_name: Name of the API key
        """
        self.auth_success_count.labels(key_name=key_name).inc()

    def record_auth_failure(self, reason: str) -> None:
        """Record authentication failure.

        Args:
            reason: Failure reason (missing_token, invalid_token, expired_token)
        """
        self.auth_failure_count.labels(reason=reason).inc()
        logger.debug(f"Auth failure recorded: {reason}")

    def record_rate_limit_hit(self, key: str, remaining: int) -> None:
        """Record rate limit hit (within limit).

        Args:
            key: Rate limit key (API key or IP)
            remaining: Remaining requests in window
        """
        self.rate_limit_hit_count.labels(key=self._sanitize_key(key)).inc()
        self.rate_limit_remaining.labels(key=self._sanitize_key(key)).set(remaining)

    def record_rate_limit_exceeded(self, key: str) -> None:
        """Record rate limit exceeded (denied request).

        Args:
            key: Rate limit key (API key or IP)
        """
        self.rate_limit_exceeded_count.labels(key=self._sanitize_key(key)).inc()
        logger.debug(f"Rate limit exceeded: {self._sanitize_key(key)}")

    def record_tool_execution(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record tool execution metrics.

        Args:
            tool_name: Name of the tool
            duration: Execution duration in seconds
            success: Whether execution succeeded
            error_type: Type of error if failed
        """
        status = "success" if success else "error"
        self.tool_execution_count.labels(tool=tool_name, status=status).inc()
        self.tool_execution_duration.labels(tool=tool_name).observe(duration)

        if not success and error_type:
            self.tool_execution_errors.labels(
                tool=tool_name,
                error_type=error_type,
            ).inc()

        logger.debug(
            f"Tool execution recorded: {tool_name} {status} {duration:.3f}s",
            extra={
                "context": {
                    "tool": tool_name,
                    "duration": duration,
                    "success": success,
                }
            },
        )

    def set_active_connections(self, count: int) -> None:
        """Set current number of active connections.

        Args:
            count: Number of active connections
        """
        self.active_connections.set(count)

    def set_queue_size(self, size: int) -> None:
        """Set current request queue size.

        Args:
            size: Number of requests in queue
        """
        self.request_queue_size.set(size)

    def set_server_info(self, name: str, version: str, environment: str) -> None:
        """Set server information.

        Args:
            name: Server name
            version: Server version
            environment: Deployment environment
        """
        self.server_info.info({
            "name": name,
            "version": version,
            "environment": environment,
        })

    def get_latest_metrics(self) -> bytes:
        """Get current metrics in Prometheus format.

        Returns:
            Metrics in Prometheus text format
        """
        if generate_latest is None:
            raise RuntimeError("Prometheus client not available")
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """Get Prometheus metrics content type.

        Returns:
            Content type string
        """
        if CONTENT_TYPE_LATEST is None:
            raise RuntimeError("Prometheus client not available")
        return CONTENT_TYPE_LATEST

    @staticmethod
    def _sanitize_key(key: str) -> str:
        """Sanitize key for metrics (truncate for privacy).

        Args:
            key: Key to sanitize

        Returns:
            Sanitized key
        """
        if len(key) > 12:
            return f"{key[:8]}..."
        return key


# Singleton instance
_metrics_instance: HttpMetrics | None = None


def get_metrics(registry: Any | None = None) -> HttpMetrics:
    """Get or create singleton metrics instance.

    Args:
        registry: Optional Prometheus registry

    Returns:
        HttpMetrics instance
    """
    global _metrics_instance

    if _metrics_instance is None:
        _metrics_instance = HttpMetrics(registry=registry)

    return _metrics_instance


class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection.

    This middleware automatically records metrics for all HTTP requests:
    - Request count, duration, size
    - Response size
    - Status codes

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> metrics = get_metrics()
        >>> @app.middleware("http")
        >>> async def metrics_middleware(request, call_next):
        >>>     middleware = MetricsMiddleware(metrics)
        >>>     return await middleware(request, call_next)
    """

    def __init__(self, metrics: HttpMetrics) -> None:
        """Initialize middleware.

        Args:
            metrics: HttpMetrics instance
        """
        self.metrics = metrics

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Process request and record metrics.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with metrics recorded
        """
        # Extract request info
        method = request.method
        path = request.url.path

        # Start timing
        start_time = time.time()

        # Get request size
        request_size = 0
        if hasattr(request, "headers") and "content-length" in request.headers:
            try:
                request_size = int(request.headers["content-length"])
            except (ValueError, TypeError):
                pass

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Get response size
        response_size = 0
        if hasattr(response, "headers") and "content-length" in response.headers:
            try:
                response_size = int(response.headers["content-length"])
            except (ValueError, TypeError):
                pass

        # Record metrics
        self.metrics.record_request(
            endpoint=path,
            method=method,
            status_code=response.status_code,
            duration=duration,
            request_size=request_size,
            response_size=response_size,
        )

        return response


__all__ = [
    "HttpMetrics",
    "MetricsMiddleware",
    "get_metrics",
]
