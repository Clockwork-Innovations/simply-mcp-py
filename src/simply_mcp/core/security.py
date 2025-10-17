"""Security features for HTTP Transport.

This module provides production-grade security features:
- Security headers (HSTS, X-Content-Type-Options, etc.)
- CORS middleware
- Request size limits
- Request timeout handling
- Input validation and sanitization
- SQL injection and path traversal prevention

All security features are configurable and can be enabled/disabled as needed.
"""

import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Type checking only - imports always succeed for static analysis
    from fastapi import Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware

    FASTAPI_AVAILABLE: bool
else:
    # Runtime behavior - handle optional FastAPI gracefully
    try:
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.middleware.cors import CORSMiddleware
        FASTAPI_AVAILABLE = True
    except ImportError:
        FASTAPI_AVAILABLE = False
        # Provide stub implementations for runtime
        Request = Any
        Response = Any
        JSONResponse = Any
        CORSMiddleware = Any
        BaseHTTPMiddleware = Any

from simply_mcp.core.logger import get_logger

logger = get_logger(__name__)

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    r"(\bUNION\b.*\bSELECT\b)",
    r"(\bINSERT\b.*\bINTO\b)",
    r"(\bDELETE\b.*\bFROM\b)",
    r"(\bUPDATE\b.*\bSET\b)",
    r"(\bDROP\b.*\bTABLE\b)",
    r"(\bCREATE\b.*\bTABLE\b)",
    r"(--|\#|\/\*|\*\/)",
    r"(\bEXEC\b|\bEXECUTE\b)",
    r"(\bxp_\w+)",
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.",
    r"%2e%2e",
    r"%252e%252e",
    r"\.\.\\",
]

# XSS patterns (basic)
XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"onerror\s*=",
    r"onload\s*=",
]


class SecurityHeadersMiddleware:
    """Middleware for adding security headers to responses.

    Adds standard security headers to all HTTP responses:
    - Strict-Transport-Security (HSTS)
    - X-Content-Type-Options
    - X-Frame-Options
    - X-XSS-Protection
    - Content-Security-Policy

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> @app.middleware("http")
        >>> async def security_headers(request, call_next):
        >>>     middleware = SecurityHeadersMiddleware(
        >>>         hsts_enabled=True,
        >>>         hsts_max_age=31536000
        >>>     )
        >>>     return await middleware(request, call_next)
    """

    def __init__(
        self,
        hsts_enabled: bool = True,
        hsts_max_age: int = 31536000,
        content_type_nosniff: bool = True,
        xss_protection: bool = True,
        frame_options: str = "DENY",
        csp_policy: str | None = None,
    ) -> None:
        """Initialize security headers middleware.

        Args:
            hsts_enabled: Enable HSTS header
            hsts_max_age: HSTS max age in seconds
            content_type_nosniff: Enable X-Content-Type-Options
            xss_protection: Enable X-XSS-Protection
            frame_options: X-Frame-Options value (DENY, SAMEORIGIN)
            csp_policy: Optional Content-Security-Policy
        """
        self.hsts_enabled = hsts_enabled
        self.hsts_max_age = hsts_max_age
        self.content_type_nosniff = content_type_nosniff
        self.xss_protection = xss_protection
        self.frame_options = frame_options
        self.csp_policy = csp_policy

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Add security headers to response.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response with security headers
        """
        response = await call_next(request)

        # Add HSTS header
        if self.hsts_enabled:
            response.headers["Strict-Transport-Security"] = (
                f"max-age={self.hsts_max_age}; includeSubDomains"
            )

        # Add X-Content-Type-Options header
        if self.content_type_nosniff:
            response.headers["X-Content-Type-Options"] = "nosniff"

        # Add X-Frame-Options header
        response.headers["X-Frame-Options"] = self.frame_options

        # Add X-XSS-Protection header
        if self.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"

        # Add Content-Security-Policy header
        if self.csp_policy:
            response.headers["Content-Security-Policy"] = self.csp_policy

        # Add X-Permitted-Cross-Domain-Policies
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"

        # Add Referrer-Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response


class RequestSizeLimitMiddleware:
    """Middleware for enforcing request size limits.

    Prevents large requests from consuming excessive memory or bandwidth.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> @app.middleware("http")
        >>> async def size_limit(request, call_next):
        >>>     middleware = RequestSizeLimitMiddleware(max_size=10*1024*1024)
        >>>     return await middleware(request, call_next)
    """

    def __init__(self, max_size: int = 10 * 1024 * 1024) -> None:
        """Initialize request size limit middleware.

        Args:
            max_size: Maximum request size in bytes (default: 10MB)
        """
        self.max_size = max_size

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Check request size and reject if too large.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response or error if request too large
        """
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size:
                    logger.warning(
                        f"Request size {size} exceeds limit {self.max_size}",
                        extra={"context": {"size": size, "limit": self.max_size}},
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "Payload Too Large",
                            "detail": f"Request size exceeds maximum of {self.max_size} bytes",
                        },
                    )
            except ValueError:
                pass

        return await call_next(request)


class RequestTimeoutMiddleware:
    """Middleware for enforcing request timeouts.

    Prevents long-running requests from blocking resources.

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> @app.middleware("http")
        >>> async def timeout(request, call_next):
        >>>     middleware = RequestTimeoutMiddleware(timeout=30)
        >>>     return await middleware(request, call_next)
    """

    def __init__(self, timeout: int = 30) -> None:
        """Initialize request timeout middleware.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Enforce request timeout.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response or timeout error
        """
        import asyncio

        start_time = time.time()

        try:
            # Process request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout,
            )
            return response

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(
                f"Request timeout after {elapsed:.2f}s",
                extra={
                    "context": {
                        "path": request.url.path,
                        "timeout": self.timeout,
                        "elapsed": elapsed,
                    }
                },
            )
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "detail": f"Request exceeded timeout of {self.timeout} seconds",
                },
            )


class InputValidationMiddleware:
    """Middleware for input validation and sanitization.

    Checks for common attack patterns:
    - SQL injection
    - Path traversal
    - XSS (basic detection)

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> @app.middleware("http")
        >>> async def input_validation(request, call_next):
        >>>     middleware = InputValidationMiddleware()
        >>>     return await middleware(request, call_next)
    """

    def __init__(
        self,
        check_sql_injection: bool = True,
        check_path_traversal: bool = True,
        check_xss: bool = True,
    ) -> None:
        """Initialize input validation middleware.

        Args:
            check_sql_injection: Enable SQL injection detection
            check_path_traversal: Enable path traversal detection
            check_xss: Enable XSS detection
        """
        self.check_sql_injection = check_sql_injection
        self.check_path_traversal = check_path_traversal
        self.check_xss = check_xss

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Validate request input.

        Args:
            request: FastAPI request
            call_next: Next middleware/handler

        Returns:
            Response or validation error
        """
        # Check URL path
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else ""

        # Check for path traversal in URL
        if self.check_path_traversal:
            if self._contains_path_traversal(path) or self._contains_path_traversal(query):
                logger.warning(
                    "Path traversal attempt detected",
                    extra={"context": {"path": path, "query": query}},
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "Bad Request",
                        "detail": "Invalid path or query parameters",
                    },
                )

        # Check request body if present
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                # Try to get body as text
                body = await request.body()
                body_text = body.decode("utf-8", errors="ignore")

                # Check for SQL injection
                if self.check_sql_injection and self._contains_sql_injection(body_text):
                    logger.warning(
                        "SQL injection attempt detected",
                        extra={"context": {"path": path}},
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Bad Request",
                            "detail": "Invalid request content",
                        },
                    )

                # Check for XSS
                if self.check_xss and self._contains_xss(body_text):
                    logger.warning(
                        "XSS attempt detected",
                        extra={"context": {"path": path}},
                    )
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": "Bad Request",
                            "detail": "Invalid request content",
                        },
                    )

                # Restore body for next handler (important!)
                async def receive() -> dict[str, Any]:
                    return {"type": "http.request", "body": body}

                request._receive = receive  # type: ignore[attr-defined]

            except Exception as e:
                logger.debug(f"Could not validate request body: {e}")

        return await call_next(request)

    @staticmethod
    def _contains_sql_injection(text: str) -> bool:
        """Check if text contains SQL injection patterns.

        Args:
            text: Text to check

        Returns:
            True if SQL injection detected
        """
        for pattern in SQL_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _contains_path_traversal(text: str) -> bool:
        """Check if text contains path traversal patterns.

        Args:
            text: Text to check

        Returns:
            True if path traversal detected
        """
        for pattern in PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _contains_xss(text: str) -> bool:
        """Check if text contains XSS patterns.

        Args:
            text: Text to check

        Returns:
            True if XSS detected
        """
        for pattern in XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


def create_cors_middleware(
    allow_origins: list[str],
    allow_methods: list[str],
    allow_headers: list[str],
    allow_credentials: bool = False,
    max_age: int = 3600,
) -> Any:
    """Create CORS middleware configuration.

    Args:
        allow_origins: List of allowed origins (* for all)
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        allow_credentials: Allow credentials
        max_age: Preflight cache duration in seconds

    Returns:
        CORSMiddleware configuration dict

    Example:
        >>> cors_config = create_cors_middleware(
        >>>     allow_origins=["https://example.com"],
        >>>     allow_methods=["GET", "POST"],
        >>>     allow_headers=["*"],
        >>> )
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for CORS middleware. "
            "Install with: pip install fastapi"
        )

    return {
        "allow_origins": allow_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": allow_methods,
        "allow_headers": allow_headers,
        "max_age": max_age,
    }


__all__ = [
    "SecurityHeadersMiddleware",
    "RequestSizeLimitMiddleware",
    "RequestTimeoutMiddleware",
    "InputValidationMiddleware",
    "create_cors_middleware",
]
