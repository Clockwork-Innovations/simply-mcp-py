"""HTTP Transport with Authentication, Rate Limiting, and Production Polish for Simply-MCP.

This module provides HTTP transport that exposes MCP tools as REST endpoints
with optional authentication, rate limiting, and production-grade features.

Features (Foundation Layer):
- Exposes MCP tools as HTTP REST endpoints
- JSON request/response handling
- Basic request validation
- Error handling with proper HTTP status codes
- Structured logging at key points
- Configurable host and port

Features (Feature Layer):
- Bearer token authentication
- API key management
- Per-key rate limiting
- Token bucket algorithm
- Rate limit headers in responses
- Backward compatible (auth is optional)

Features (Polish Layer):
- YAML/TOML configuration file support
- Environment variable configuration
- Prometheus metrics collection
- Security headers and CORS
- HTTPS/TLS support
- Request size limits and timeouts
- Input validation (SQL injection, path traversal, XSS)
- Graceful shutdown with connection draining
- Health check with component status
- Structured JSON logging
"""

import asyncio
import logging
import signal
import uuid
from typing import TYPE_CHECKING, Any

# Foundation layer dependencies (FastAPI, uvicorn)
# Separated for type checking to eliminate type: ignore statements
if TYPE_CHECKING:
    # Type checking only - imports always succeed for static analysis
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, Response, status
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.middleware.cors import CORSMiddleware
else:
    # Runtime behavior - handle optional dependencies gracefully
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Request, Response, status
        from fastapi.responses import JSONResponse, PlainTextResponse
        from starlette.middleware.cors import CORSMiddleware

        FASTAPI_AVAILABLE = True
    except ImportError:
        FASTAPI_AVAILABLE = False
        # Provide stub implementations for runtime when dependencies missing
        FastAPI = Any
        HTTPException = Exception
        Request = Any
        Response = Any
        status = Any
        JSONResponse = Any
        PlainTextResponse = Any
        CORSMiddleware = Any
        uvicorn = Any

# Polish layer dependencies (configuration, metrics, security)
# Separated for type checking to eliminate type: ignore statements
if TYPE_CHECKING:
    # Type checking only - imports always succeed for static analysis
    from simply_mcp.core.http_config import HttpConfig
    from simply_mcp.core.security import (
        InputValidationMiddleware,
        RequestSizeLimitMiddleware,
        RequestTimeoutMiddleware,
        SecurityHeadersMiddleware,
    )
    from simply_mcp.monitoring.http_metrics import HttpMetrics, get_metrics
else:
    # Runtime behavior - handle optional polish layer gracefully
    try:
        from simply_mcp.core.http_config import HttpConfig
        from simply_mcp.core.security import (
            InputValidationMiddleware,
            RequestSizeLimitMiddleware,
            RequestTimeoutMiddleware,
            SecurityHeadersMiddleware,
        )
        from simply_mcp.monitoring.http_metrics import HttpMetrics, get_metrics

        POLISH_LAYER_AVAILABLE = True
    except ImportError:
        POLISH_LAYER_AVAILABLE = False
        # Provide stub implementations for runtime when dependencies missing
        HttpConfig = Any
        HttpMetrics = Any
        get_metrics = None
        SecurityHeadersMiddleware = Any
        RequestSizeLimitMiddleware = Any
        RequestTimeoutMiddleware = Any
        InputValidationMiddleware = Any

from simply_mcp.core.auth import ApiKey, ApiKeyManager, BearerTokenValidator
from simply_mcp.core.logger import LoggerContext, get_logger
from simply_mcp.core.rate_limit import RateLimitConfig, RateLimiter

logger = get_logger(__name__)


class HttpTransport:
    """HTTP transport for exposing MCP tools as REST endpoints.

    This is the foundation layer implementation - it provides basic HTTP
    server functionality to expose MCP tools via REST endpoints.

    Args:
        server: BuildMCPServer instance with registered tools
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8000)

    Example:
        >>> from simply_mcp import BuildMCPServer
        >>> from simply_mcp.transports.http_transport import HttpTransport
        >>>
        >>> # Create and configure server
        >>> mcp = BuildMCPServer(name="my-server", version="1.0.0")
        >>> # ... register tools ...
        >>>
        >>> # Create HTTP transport
        >>> transport = HttpTransport(server=mcp, host="0.0.0.0", port=8000)
        >>> await transport.start()
    """

    def __init__(
        self,
        server: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_auth: bool = False,
        api_keys: list[ApiKey] | None = None,
        enable_rate_limiting: bool = False,
        config: Any | None = None,
    ) -> None:
        """Initialize HTTP transport.

        Args:
            server: BuildMCPServer instance with registered tools
            host: Host to bind to (ignored if config provided)
            port: Port to bind to (ignored if config provided)
            enable_auth: Enable authentication (ignored if config provided)
            api_keys: List of ApiKey objects for authentication
            enable_rate_limiting: Enable rate limiting (ignored if config provided)
            config: Optional HttpConfig instance for production features

        Raises:
            ImportError: If FastAPI or uvicorn are not installed
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI and uvicorn are required for HTTP transport. "
                "Install with: pip install fastapi uvicorn"
            )

        self.server = server
        self.config = config
        self.app: Any = None
        self._server_task: Any = None
        self._uvicorn_server: Any = None
        self._shutdown_event = asyncio.Event()

        # Apply configuration or use direct parameters
        if config:
            self.host = config.server.host
            self.port = config.server.port
            self.enable_auth = config.auth.enabled
            self.enable_rate_limiting = config.rate_limit.enabled
        else:
            self.host = host
            self.port = port
            self.enable_auth = enable_auth
            self.enable_rate_limiting = enable_rate_limiting

        # Initialize authentication manager
        self.api_key_manager = ApiKeyManager()
        if api_keys:
            for api_key in api_keys:
                self.api_key_manager.add_key(api_key)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter()

        # Set up rate limiting for keys if both auth and rate limiting are enabled
        if self.enable_auth and self.enable_rate_limiting and api_keys:
            for api_key in api_keys:
                rate_config = RateLimitConfig(
                    max_requests=api_key.rate_limit,
                    window_seconds=api_key.window_seconds,
                )
                self.rate_limiter.add_key(api_key.key, rate_config)

        # Initialize metrics if polish layer available and enabled
        self.metrics: Any | None = None
        if POLISH_LAYER_AVAILABLE and config and config.monitoring.prometheus_enabled:
            if get_metrics is not None:
                self.metrics = get_metrics()
                self.metrics.set_server_info(
                    name=server.name,
                    version=server.version,
                    environment=config.environment,
                )

        logger.info(
            f"HTTP transport initialized: {self.host}:{self.port}",
            extra={
                "context": {
                    "host": self.host,
                    "port": self.port,
                    "auth_enabled": self.enable_auth,
                    "rate_limiting_enabled": self.enable_rate_limiting,
                    "api_keys_count": len(api_keys) if api_keys else 0,
                    "polish_layer": POLISH_LAYER_AVAILABLE and config is not None,
                }
            },
        )

    def _create_app(self) -> Any:
        """Create FastAPI application with tool endpoints.

        Returns:
            FastAPI application instance
        """
        app = FastAPI(
            title=self.server.name,
            version=self.server.version,
            description=self.server.description or "MCP Server HTTP Interface",
        )

        # Add CORS middleware if polish layer enabled
        if POLISH_LAYER_AVAILABLE and self.config and self.config.cors.enabled and CORSMiddleware is not None:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors.allow_origins,
                allow_credentials=self.config.cors.allow_credentials,
                allow_methods=self.config.cors.allow_methods,
                allow_headers=self.config.cors.allow_headers,
                max_age=self.config.cors.max_age,
            )
            logger.info("CORS middleware enabled")

        # Add security middleware if polish layer enabled
        if POLISH_LAYER_AVAILABLE and self.config:
            # Security headers
            if self.config.security.security_headers and SecurityHeadersMiddleware is not None:
                @app.middleware("http")
                async def security_headers_middleware(request: Request, call_next: Any) -> Response:
                    middleware = SecurityHeadersMiddleware(
                        hsts_enabled=self.config.security.hsts_enabled,
                        hsts_max_age=self.config.security.hsts_max_age,
                        content_type_nosniff=self.config.security.content_type_nosniff,
                        xss_protection=self.config.security.xss_protection,
                        frame_options=self.config.security.frame_options,
                    )
                    return await middleware(request, call_next)
                logger.info("Security headers middleware enabled")

            # Request size limit
            if RequestSizeLimitMiddleware is not None:
                @app.middleware("http")
                async def size_limit_middleware(request: Request, call_next: Any) -> Response:
                    middleware = RequestSizeLimitMiddleware(
                        max_size=self.config.security.max_request_size
                    )
                    return await middleware(request, call_next)

            # Request timeout
            if RequestTimeoutMiddleware is not None:
                @app.middleware("http")
                async def timeout_middleware(request: Request, call_next: Any) -> Response:
                    middleware = RequestTimeoutMiddleware(
                        timeout=self.config.security.request_timeout
                    )
                    return await middleware(request, call_next)

            # Input validation
            if InputValidationMiddleware is not None:
                @app.middleware("http")
                async def input_validation_middleware(request: Request, call_next: Any) -> Response:
                    middleware = InputValidationMiddleware()
                    return await middleware(request, call_next)

        # Add metrics middleware if polish layer enabled
        if self.metrics:
            @app.middleware("http")
            async def metrics_middleware(request: Request, call_next: Any) -> Response:
                import time

                # Add correlation ID
                correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
                request.state.correlation_id = correlation_id

                # Start timing
                start_time = time.time()

                # Get request size
                request_size = 0
                if "content-length" in request.headers:
                    try:
                        request_size = int(request.headers["content-length"])
                    except (ValueError, TypeError):
                        pass

                # Log request with correlation ID
                with LoggerContext(correlation_id=correlation_id):
                    client_host = request.client.host if request.client is not None else "unknown"
                    logger.info(
                        f"{request.method} {request.url.path}",
                        extra={
                            "context": {
                                "method": request.method,
                                "path": request.url.path,
                                "client": client_host,
                            }
                        },
                    )

                    # Process request
                    response = await call_next(request)

                    # Calculate duration
                    duration = time.time() - start_time

                    # Get response size
                    response_size = 0
                    if "content-length" in response.headers:
                        try:
                            response_size = int(response.headers["content-length"])
                        except (ValueError, TypeError):
                            pass

                    # Record metrics
                    self.metrics.record_request(
                        endpoint=request.url.path,
                        method=request.method,
                        status_code=response.status_code,
                        duration=duration,
                        request_size=request_size,
                        response_size=response_size,
                    )

                    # Add correlation ID to response
                    response.headers["X-Correlation-ID"] = correlation_id

                    return response
            logger.info("Metrics middleware enabled")

        # Add authentication and rate limiting middleware
        if self.enable_auth or self.enable_rate_limiting:
            @app.middleware("http")
            async def auth_and_rate_limit_middleware(request: Request, call_next: Any) -> Response:
                """Middleware for authentication and rate limiting."""
                # Skip middleware for health and metrics endpoints
                if request.url.path in ("/health", "/metrics"):
                    return await call_next(request)

                # Store request context
                request.state.api_key = None
                request.state.rate_limit_info = None

                # Authentication check
                if self.enable_auth:
                    auth_header = request.headers.get("Authorization")
                    token = BearerTokenValidator.extract_token(auth_header)

                    if not token:
                        logger.warning(
                            "Unauthorized request - missing or invalid token",
                            extra={"path": request.url.path, "method": request.method}
                        )
                        if self.metrics:
                            self.metrics.record_auth_failure("missing_token")
                        return JSONResponse(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            content={
                                "error": "Unauthorized",
                                "detail": "Missing or invalid Authorization header. Expected format: 'Bearer <token>'",
                            },
                            headers={"WWW-Authenticate": "Bearer"},
                        )

                    # Validate token
                    is_valid, api_key = self.api_key_manager.validate_token(token)
                    if not is_valid:
                        logger.warning(
                            "Unauthorized request - invalid API key",
                            extra={"path": request.url.path, "method": request.method}
                        )
                        if self.metrics:
                            self.metrics.record_auth_failure("invalid_token")
                        return JSONResponse(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            content={
                                "error": "Unauthorized",
                                "detail": "Invalid API key",
                            },
                            headers={"WWW-Authenticate": "Bearer"},
                        )

                    # Store authenticated key in request state
                    request.state.api_key = api_key
                    if self.metrics:
                        self.metrics.record_auth_success(api_key.name)
                    logger.debug(
                        f"Request authenticated: {api_key.name}",
                        extra={"key_name": api_key.name, "path": request.url.path}
                    )

                # Rate limiting check
                if self.enable_rate_limiting:
                    # Get API key for rate limiting
                    key_for_limit = None
                    if self.enable_auth and request.state.api_key:
                        key_for_limit = request.state.api_key.key
                    elif not self.enable_auth:
                        # If auth is disabled, use IP address for rate limiting
                        key_for_limit = request.client.host if request.client is not None else "unknown"

                    if key_for_limit:
                        allowed, rate_info = self.rate_limiter.check_limit(key_for_limit)
                        request.state.rate_limit_info = rate_info

                        if not allowed:
                            logger.warning(
                                "Rate limit exceeded",
                                extra={
                                    "key_prefix": key_for_limit[:8] + "..." if len(key_for_limit) > 8 else key_for_limit,
                                    "path": request.url.path,
                                    "retry_after": rate_info.retry_after,
                                }
                            )
                            if self.metrics:
                                self.metrics.record_rate_limit_exceeded(key_for_limit)
                            return JSONResponse(
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                content={
                                    "error": "Rate limit exceeded",
                                    "detail": f"Too many requests. Retry after {rate_info.retry_after:.1f} seconds",
                                    "retry_after": rate_info.retry_after,
                                },
                                headers={
                                    "X-RateLimit-Limit": str(rate_info.limit),
                                    "X-RateLimit-Remaining": str(rate_info.remaining),
                                    "X-RateLimit-Reset": str(int(rate_info.reset_at)),
                                    "Retry-After": str(int(rate_info.retry_after) + 1),
                                },
                            )
                        elif self.metrics:
                            self.metrics.record_rate_limit_hit(key_for_limit, rate_info.remaining)

                # Process request
                response = await call_next(request)

                # Add rate limit headers to response if available
                if self.enable_rate_limiting and hasattr(request.state, "rate_limit_info"):
                    rate_info = request.state.rate_limit_info
                    if rate_info is not None:
                        response.headers["X-RateLimit-Limit"] = str(rate_info.limit)
                        response.headers["X-RateLimit-Remaining"] = str(rate_info.remaining)
                        response.headers["X-RateLimit-Reset"] = str(int(rate_info.reset_at))

                return response

        # Enhanced health check endpoint
        @app.get("/health")
        async def health_check() -> dict[str, Any]:
            """Enhanced health check endpoint with component status."""
            logger.debug("Health check request received")

            health_data: dict[str, Any] = {
                "status": "healthy",
                "server": self.server.name,
                "version": self.server.version,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Add component status if polish layer enabled
            if self.config:
                components: dict[str, dict[str, Any]] = {}

                # Server component
                components["server"] = {
                    "status": "up",
                    "host": self.host,
                    "port": self.port,
                    "environment": self.config.environment,
                }

                # Auth component
                if self.enable_auth:
                    components["authentication"] = {
                        "status": "enabled",
                        "keys_count": len(self.api_key_manager.keys),
                    }

                # Rate limiting component
                if self.enable_rate_limiting:
                    strategy = self.config.rate_limit.strategy if self.config is not None else "token_bucket"
                    components["rate_limiting"] = {
                        "status": "enabled",
                        "strategy": strategy,
                    }

                # Metrics component
                if self.metrics and self.config is not None:
                    components["metrics"] = {
                        "status": "enabled",
                        "endpoint": self.config.monitoring.prometheus_path,
                    }

                # Security component
                if self.config.security.security_headers:
                    components["security"] = {
                        "status": "enabled",
                        "hsts": self.config.security.hsts_enabled,
                        "cors": self.config.cors.enabled,
                    }

                health_data["components"] = components

            return health_data

        # Metrics endpoint (if metrics enabled)
        if self.metrics and PlainTextResponse is not None:
            metrics_path = self.config.monitoring.prometheus_path if self.config else "/metrics"
            @app.get(metrics_path)
            async def metrics_endpoint() -> Response:
                """Prometheus metrics endpoint."""
                logger.debug("Metrics request received")
                if self.metrics is None:
                    raise HTTPException(status_code=500, detail="Metrics not available")
                metrics_data = self.metrics.get_latest_metrics()
                return PlainTextResponse(
                    content=metrics_data,
                    media_type=self.metrics.get_content_type(),
                )

        # List all available tools
        @app.get("/tools")
        async def list_tools() -> dict[str, Any]:
            """List all available tools."""
            logger.info("Listing tools via HTTP")
            try:
                tools = self.server.list_tools()
                logger.debug(f"Listed {len(tools)} tools")
                return {"tools": tools, "count": len(tools)}
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to list tools: {str(e)}",
                ) from e

        # Call a specific tool (generic endpoint)
        @app.post("/tools/{tool_name}")
        async def call_tool(tool_name: str, request: Request) -> Response:
            """Call a specific tool with JSON parameters.

            Args:
                tool_name: Name of the tool to call
                request: HTTP request containing JSON parameters

            Returns:
                Tool execution result as JSON response
            """
            logger.info(f"HTTP request to call tool: {tool_name}")

            try:
                # Parse request body as JSON
                try:
                    body = await request.json()
                except Exception as e:
                    logger.warning(f"Invalid JSON in request body: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid JSON in request body: {str(e)}",
                    ) from e

                # Get tool configuration from registry
                tool_config = self.server.server.registry.get_tool(tool_name)
                if not tool_config:
                    logger.warning(f"Tool not found: {tool_name}")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Tool not found: {tool_name}",
                    )

                # Extract parameters from body
                params = body if isinstance(body, dict) else {}

                # Execute tool handler
                logger.debug(f"Executing tool: {tool_name} with params: {params}")

                # Track execution time
                import time
                tool_start = time.time()
                tool_error_type: str | None = None

                try:
                    result = tool_config.handler(**params)

                    # Handle async handlers
                    if asyncio.iscoroutine(result):
                        result = await result

                    logger.info(f"Tool executed successfully: {tool_name}")

                    # Record tool execution metrics
                    if self.metrics:
                        tool_duration = time.time() - tool_start
                        self.metrics.record_tool_execution(
                            tool_name=tool_name,
                            duration=tool_duration,
                            success=True,
                        )

                    # Return result as JSON
                    return JSONResponse(
                        status_code=status.HTTP_200_OK,
                        content={
                            "success": True,
                            "tool": tool_name,
                            "result": result,
                        },
                    )

                except TypeError as e:
                    # Parameter validation error
                    tool_error_type = "parameter_error"
                    logger.warning(f"Invalid parameters for tool {tool_name}: {e}")

                    # Record tool execution error
                    if self.metrics:
                        tool_duration = time.time() - tool_start
                        self.metrics.record_tool_execution(
                            tool_name=tool_name,
                            duration=tool_duration,
                            success=False,
                            error_type=tool_error_type,
                        )

                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid parameters: {str(e)}",
                    ) from e

                except Exception as e:
                    # Tool execution error
                    tool_error_type = "execution_error"
                    logger.error(f"Tool execution failed: {tool_name} - {e}")

                    # Record tool execution error
                    if self.metrics:
                        tool_duration = time.time() - tool_start
                        self.metrics.record_tool_execution(
                            tool_name=tool_name,
                            duration=tool_duration,
                            success=False,
                            error_type=tool_error_type,
                        )

                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Tool execution failed: {str(e)}",
                    ) from e

            except HTTPException:
                # Re-raise HTTP exceptions
                raise

            except Exception as e:
                # Catch-all for unexpected errors
                logger.error(f"Unexpected error handling request: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {str(e)}",
                ) from e

        # Add specific endpoints for each registered tool
        self._register_tool_endpoints(app)

        logger.info("FastAPI app created with tool endpoints")
        return app

    def _register_tool_endpoints(self, app: Any) -> None:
        """Register specific endpoints for each tool.

        This creates dedicated endpoints for each tool to provide
        better API ergonomics alongside the generic /tools/{name} endpoint.

        Args:
            app: FastAPI application instance
        """
        tools = self.server.list_tools()
        logger.debug(f"Registering {len(tools)} tool-specific endpoints")

        for tool_name in tools:
            # Create endpoint at /api/{tool_name}
            endpoint_path = f"/api/{tool_name}"

            # We need to capture tool_name in the closure
            def make_endpoint(name: str) -> Any:
                async def endpoint(request: Request) -> Response:
                    """Tool-specific endpoint."""
                    # Forward to generic tool handler
                    return await app.router.routes[-1].endpoint(name, request)

                return endpoint

            # Register the endpoint
            app.post(endpoint_path)(make_endpoint(tool_name))
            logger.debug(f"Registered endpoint: POST {endpoint_path}")

    async def start(self) -> None:
        """Start the HTTP server.

        This initializes the FastAPI app and starts the uvicorn server
        in the background. Supports HTTPS/TLS if configured.

        Raises:
            RuntimeError: If server is already running
        """
        if self._server_task is not None:
            raise RuntimeError("HTTP transport is already running")

        logger.info(f"Starting HTTP transport on {self.host}:{self.port}")

        # Create FastAPI app
        self.app = self._create_app()

        # Determine TLS configuration
        ssl_keyfile: str | None = None
        ssl_certfile: str | None = None
        ssl_ca_certs: str | None = None

        if self.config is not None and self.config.tls.enabled:
            ssl_certfile = self.config.tls.cert_file
            ssl_keyfile = self.config.tls.key_file
            ssl_ca_certs = self.config.tls.ca_file
            logger.info("HTTPS/TLS enabled")

        # Create uvicorn server config
        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level=logging.WARNING,  # Reduce uvicorn noise
            access_log=False,  # We log in our handlers
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            ssl_ca_certs=ssl_ca_certs,
            workers=1,  # Single worker for now
        )

        # Create server instance
        self._uvicorn_server = uvicorn.Server(config)

        # Setup graceful shutdown handler
        self._setup_shutdown_handlers()

        # Start server in background task
        self._server_task = asyncio.create_task(self._uvicorn_server.serve())

        # Give server time to start
        await asyncio.sleep(0.5)

        protocol = "https" if ssl_certfile else "http"
        logger.info(
            f"HTTP transport started successfully on {protocol}://{self.host}:{self.port}",
            extra={
                "context": {
                    "host": self.host,
                    "port": self.port,
                    "protocol": protocol,
                    "endpoint": f"{protocol}://{self.host}:{self.port}",
                    "tls_enabled": ssl_certfile is not None,
                }
            },
        )

    def _setup_shutdown_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        def signal_handler(sig: int, frame: Any) -> None:
            """Handle shutdown signals."""
            logger.info(f"Received signal {sig}, initiating graceful shutdown")
            self._shutdown_event.set()

            # Trigger uvicorn shutdown
            if self._uvicorn_server:
                self._uvicorn_server.should_exit = True

        # Register signal handlers (only on Unix-like systems)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal_handler)

    async def stop(self) -> None:
        """Stop the HTTP server gracefully.

        This shuts down the uvicorn server and cleans up resources with
        graceful connection draining.
        """
        if self._server_task is None:
            logger.warning("HTTP transport is not running")
            return

        logger.info("Stopping HTTP transport gracefully")

        # Determine graceful timeout from config
        graceful_timeout = 30.0
        if self.config is not None:
            graceful_timeout = float(self.config.server.graceful_timeout)

        try:
            # Signal uvicorn to shutdown
            if self._uvicorn_server:
                self._uvicorn_server.should_exit = True
                logger.info(f"Draining connections (timeout: {graceful_timeout}s)")

            # Wait for server task to complete (with graceful timeout)
            try:
                await asyncio.wait_for(self._server_task, timeout=graceful_timeout)
                logger.info("All connections drained successfully")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Graceful shutdown timed out after {graceful_timeout}s, forcing shutdown"
                )
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    logger.info("Server task cancelled")

            logger.info("HTTP transport stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping HTTP transport: {e}", exc_info=True)
            raise

        finally:
            self._server_task = None
            self._uvicorn_server = None
            self._shutdown_event.clear()

    async def __aenter__(self) -> "HttpTransport":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.stop()


__all__ = ["HttpTransport"]
