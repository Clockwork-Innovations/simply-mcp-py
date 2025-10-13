"""Middleware for HTTP and SSE transports.

This module provides middleware components for request processing in
HTTP and SSE transports, including CORS support, logging, and rate limiting.
"""

import time
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web

from simply_mcp.core.logger import get_logger

logger = get_logger(__name__)

# Type alias for middleware handler
Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


class CORSMiddleware:
    """CORS middleware for HTTP/SSE transports.

    Handles Cross-Origin Resource Sharing (CORS) headers to allow
    web browsers to make requests to the MCP server from different origins.

    Attributes:
        enabled: Whether CORS is enabled
        allowed_origins: List of allowed origins or ["*"] for all
        allow_credentials: Whether to allow credentials
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        max_age: Maximum age for preflight cache (seconds)

    Example:
        >>> middleware = CORSMiddleware(
        ...     enabled=True,
        ...     allowed_origins=["http://localhost:3000"]
        ... )
        >>> app.middlewares.append(middleware)
    """

    def __init__(
        self,
        enabled: bool = True,
        allowed_origins: list[str] | None = None,
        allow_credentials: bool = True,
        allow_methods: list[str] | None = None,
        allow_headers: list[str] | None = None,
        max_age: int = 86400,
    ) -> None:
        """Initialize CORS middleware.

        Args:
            enabled: Whether CORS is enabled
            allowed_origins: List of allowed origins or None for ["*"]
            allow_credentials: Whether to allow credentials
            allow_methods: List of allowed methods or None for defaults
            allow_headers: List of allowed headers or None for defaults
            max_age: Maximum age for preflight cache (seconds)
        """
        self.enabled = enabled
        self.allowed_origins = allowed_origins or ["*"]
        self.allow_credentials = allow_credentials
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allow_headers = allow_headers or [
            "Content-Type",
            "Authorization",
            "X-Requested-With",
        ]
        self.max_age = max_age

    @web.middleware
    async def __call__(
        self,
        request: web.Request,
        handler: Handler,
    ) -> web.StreamResponse:
        """Process request with CORS headers.

        Args:
            request: Incoming request
            handler: Next handler in chain

        Returns:
            Response with CORS headers
        """
        if not self.enabled:
            return await handler(request)

        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            return await self._handle_preflight(request)

        # Process normal request
        response = await handler(request)

        # Add CORS headers to response
        self._add_cors_headers(request, response)

        return response

    async def _handle_preflight(self, request: web.Request) -> web.Response:
        """Handle CORS preflight OPTIONS request.

        Args:
            request: OPTIONS request

        Returns:
            Response with CORS preflight headers
        """
        response = web.Response(status=204)
        self._add_cors_headers(request, response)
        return response

    def _add_cors_headers(
        self,
        request: web.Request,
        response: web.StreamResponse,
    ) -> None:
        """Add CORS headers to response.

        Args:
            request: Incoming request
            response: Response to add headers to
        """
        origin = request.headers.get("Origin")

        # Check if origin is allowed
        if origin and (
            "*" in self.allowed_origins or origin in self.allowed_origins
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
        elif "*" in self.allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = "*"

        # Add other CORS headers
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"

        response.headers["Access-Control-Allow-Methods"] = ", ".join(
            self.allow_methods
        )
        response.headers["Access-Control-Allow-Headers"] = ", ".join(
            self.allow_headers
        )
        response.headers["Access-Control-Max-Age"] = str(self.max_age)


class LoggingMiddleware:
    """Request/response logging middleware.

    Logs HTTP requests and responses with timing information for
    debugging and monitoring purposes.

    Example:
        >>> middleware = LoggingMiddleware()
        >>> app.middlewares.append(middleware)
    """

    def __init__(self, verbose: bool = False) -> None:
        """Initialize logging middleware.

        Args:
            verbose: Whether to log request/response bodies
        """
        self.verbose = verbose

    @web.middleware
    async def __call__(
        self,
        request: web.Request,
        handler: Handler,
    ) -> web.StreamResponse:
        """Process request with logging.

        Args:
            request: Incoming request
            handler: Next handler in chain

        Returns:
            Response from handler
        """
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.path}",
            extra={
                "context": {
                    "method": request.method,
                    "path": request.path,
                    "remote": request.remote or "unknown",
                    "headers": dict(request.headers) if self.verbose else None,
                }
            },
        )

        try:
            # Process request
            response = await handler(request)

            # Calculate elapsed time
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Log response
            logger.info(
                f"Response: {response.status} ({elapsed_ms}ms)",
                extra={
                    "context": {
                        "method": request.method,
                        "path": request.path,
                        "status": response.status,
                        "elapsed_ms": elapsed_ms,
                    }
                },
            )

            return response

        except Exception as e:
            # Calculate elapsed time
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            # Log error
            logger.error(
                f"Request failed: {e} ({elapsed_ms}ms)",
                extra={
                    "context": {
                        "method": request.method,
                        "path": request.path,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "elapsed_ms": elapsed_ms,
                    }
                },
            )

            raise


class RateLimitMiddleware:
    """Rate limiting middleware.

    Provides basic rate limiting to prevent abuse. This is a simple
    implementation that can be extended in Phase 4 with more sophisticated
    rate limiting strategies.

    Note:
        This is a placeholder implementation for Phase 4 integration.
        Currently, it just logs requests without enforcing limits.

    Example:
        >>> middleware = RateLimitMiddleware(
        ...     max_requests=100,
        ...     window_seconds=60
        ... )
        >>> app.middlewares.append(middleware)
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        """Initialize rate limit middleware.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    @web.middleware
    async def __call__(
        self,
        request: web.Request,
        handler: Handler,
    ) -> web.StreamResponse:
        """Process request with rate limiting.

        Args:
            request: Incoming request
            handler: Next handler in chain

        Returns:
            Response from handler or 429 if rate limited
        """
        # Get client identifier (IP address)
        client_id = request.remote or "unknown"

        # Track request
        current_time = time.time()

        # Initialize client tracking if needed
        if client_id not in self._requests:
            self._requests[client_id] = []

        # Remove old requests outside window
        self._requests[client_id] = [
            req_time
            for req_time in self._requests[client_id]
            if current_time - req_time < self.window_seconds
        ]

        # Check rate limit (placeholder - Phase 4 will enforce)
        if len(self._requests[client_id]) >= self.max_requests:
            logger.warning(
                f"Rate limit would be exceeded for {client_id} "
                f"({len(self._requests[client_id])} requests)",
                extra={
                    "context": {
                        "client_id": client_id,
                        "request_count": len(self._requests[client_id]),
                        "max_requests": self.max_requests,
                        "window_seconds": self.window_seconds,
                    }
                },
            )
            # Phase 4: Uncomment to enforce
            # return web.Response(
            #     status=429,
            #     text="Rate limit exceeded",
            # )

        # Record this request
        self._requests[client_id].append(current_time)

        # Process request
        return await handler(request)


def create_middleware_stack(
    cors_enabled: bool = True,
    cors_origins: list[str] | None = None,
    logging_enabled: bool = True,
    rate_limit_enabled: bool = False,
) -> list[Any]:
    """Create a middleware stack for HTTP/SSE transports.

    This is a convenience function that creates commonly used middleware
    configurations.

    Args:
        cors_enabled: Whether to enable CORS middleware
        cors_origins: Allowed CORS origins or None for all (*)
        logging_enabled: Whether to enable logging middleware
        rate_limit_enabled: Whether to enable rate limiting

    Returns:
        List of middleware instances

    Example:
        >>> middlewares = create_middleware_stack(
        ...     cors_enabled=True,
        ...     cors_origins=["http://localhost:3000"],
        ...     logging_enabled=True,
        ... )
        >>> app = web.Application(middlewares=middlewares)
    """
    middlewares: list[Any] = []

    # Add logging first for complete request/response tracking
    if logging_enabled:
        middlewares.append(LoggingMiddleware())

    # Add CORS support
    if cors_enabled:
        middlewares.append(
            CORSMiddleware(
                enabled=True,
                allowed_origins=cors_origins,
            )
        )

    # Add rate limiting (placeholder for Phase 4)
    if rate_limit_enabled:
        middlewares.append(RateLimitMiddleware())

    return middlewares


__all__ = [
    "CORSMiddleware",
    "LoggingMiddleware",
    "RateLimitMiddleware",
    "create_middleware_stack",
]
