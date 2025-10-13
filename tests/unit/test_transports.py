"""Unit tests for transport implementations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from simply_mcp.core.config import get_default_config
from simply_mcp.core.server import SimplyMCPServer
from simply_mcp.transports.factory import create_transport
from simply_mcp.transports.http import HTTPTransport
from simply_mcp.transports.middleware import (
    CORSMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    create_middleware_stack,
)
from simply_mcp.transports.sse import SSEConnection, SSETransport


class TestCORSMiddleware:
    """Tests for CORS middleware."""

    def test_init_defaults(self) -> None:
        """Test CORS middleware initialization with defaults."""
        middleware = CORSMiddleware()

        assert middleware.enabled is True
        assert middleware.allowed_origins == ["*"]
        assert middleware.allow_credentials is True
        assert "GET" in middleware.allow_methods
        assert "POST" in middleware.allow_methods
        assert "Content-Type" in middleware.allow_headers

    def test_init_custom(self) -> None:
        """Test CORS middleware initialization with custom values."""
        middleware = CORSMiddleware(
            enabled=True,
            allowed_origins=["http://localhost:3000"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
            max_age=3600,
        )

        assert middleware.enabled is True
        assert middleware.allowed_origins == ["http://localhost:3000"]
        assert middleware.allow_credentials is False
        assert middleware.allow_methods == ["GET", "POST"]
        assert middleware.allow_headers == ["Content-Type"]
        assert middleware.max_age == 3600

    @pytest.mark.asyncio
    async def test_middleware_disabled(self) -> None:
        """Test CORS middleware when disabled."""
        middleware = CORSMiddleware(enabled=False)

        request = MagicMock(spec=web.Request)
        request.method = "GET"

        response = web.Response()
        handler = AsyncMock(return_value=response)

        result = await middleware(request, handler)

        assert result == response
        handler.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_middleware_adds_cors_headers(self) -> None:
        """Test CORS middleware adds appropriate headers."""
        middleware = CORSMiddleware(
            enabled=True,
            allowed_origins=["http://localhost:3000"],
        )

        request = MagicMock(spec=web.Request)
        request.method = "GET"
        request.headers = {"Origin": "http://localhost:3000"}

        response = web.Response()
        handler = AsyncMock(return_value=response)

        result = await middleware(request, handler)

        assert "Access-Control-Allow-Origin" in result.headers
        assert result.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"


class TestLoggingMiddleware:
    """Tests for logging middleware."""

    def test_init_defaults(self) -> None:
        """Test logging middleware initialization."""
        middleware = LoggingMiddleware()
        assert middleware.verbose is False

    def test_init_verbose(self) -> None:
        """Test logging middleware with verbose mode."""
        middleware = LoggingMiddleware(verbose=True)
        assert middleware.verbose is True

    @pytest.mark.asyncio
    async def test_middleware_logs_request(self) -> None:
        """Test logging middleware logs requests."""
        middleware = LoggingMiddleware()

        request = MagicMock(spec=web.Request)
        request.method = "GET"
        request.path = "/test"
        request.remote = "127.0.0.1"
        request.headers = {}

        response = web.Response(status=200)
        handler = AsyncMock(return_value=response)

        result = await middleware(request, handler)

        assert result == response
        handler.assert_called_once_with(request)


class TestRateLimitMiddleware:
    """Tests for rate limit middleware."""

    def test_init_defaults(self) -> None:
        """Test rate limit middleware initialization."""
        middleware = RateLimitMiddleware()

        assert middleware.max_requests == 100
        assert middleware.window_seconds == 60

    def test_init_custom(self) -> None:
        """Test rate limit middleware with custom values."""
        middleware = RateLimitMiddleware(
            max_requests=50,
            window_seconds=30,
        )

        assert middleware.max_requests == 50
        assert middleware.window_seconds == 30

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests(self) -> None:
        """Test rate limit middleware tracks requests."""
        middleware = RateLimitMiddleware(max_requests=10)

        request = MagicMock(spec=web.Request)
        request.remote = "127.0.0.1"

        response = web.Response()
        handler = AsyncMock(return_value=response)

        # Make a request
        result = await middleware(request, handler)

        assert result == response
        assert "127.0.0.1" in middleware._requests
        assert len(middleware._requests["127.0.0.1"]) == 1


class TestMiddlewareStack:
    """Tests for middleware stack creation."""

    def test_create_middleware_stack_defaults(self) -> None:
        """Test creating middleware stack with defaults."""
        middlewares = create_middleware_stack()

        assert len(middlewares) >= 2  # At least logging and CORS
        assert isinstance(middlewares[0], LoggingMiddleware)
        assert isinstance(middlewares[1], CORSMiddleware)

    def test_create_middleware_stack_custom(self) -> None:
        """Test creating middleware stack with custom options."""
        middlewares = create_middleware_stack(
            cors_enabled=True,
            cors_origins=["http://localhost:3000"],
            logging_enabled=True,
            rate_limit_enabled=True,
        )

        assert len(middlewares) >= 3
        assert any(isinstance(m, LoggingMiddleware) for m in middlewares)
        assert any(isinstance(m, CORSMiddleware) for m in middlewares)
        assert any(isinstance(m, RateLimitMiddleware) for m in middlewares)

    def test_create_middleware_stack_minimal(self) -> None:
        """Test creating minimal middleware stack."""
        middlewares = create_middleware_stack(
            cors_enabled=False,
            logging_enabled=False,
            rate_limit_enabled=False,
        )

        assert len(middlewares) == 0


class TestHTTPTransport:
    """Tests for HTTP transport."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test HTTP transport initialization."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        transport = HTTPTransport(
            server=server,
            host="localhost",
            port=8080,
            cors_enabled=True,
            cors_origins=["http://localhost:3000"],
        )

        assert transport.server == server
        assert transport.host == "localhost"
        assert transport.port == 8080
        assert transport.cors_enabled is True
        assert transport.cors_origins == ["http://localhost:3000"]
        assert transport.app is None
        assert transport.runner is None

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test HTTP transport start and stop."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(
            server=server,
            host="localhost",
            port=0,  # Random port
        )

        # Start transport
        await transport.start()
        assert transport.app is not None
        assert transport.runner is not None

        # Stop transport
        await transport.stop()
        assert transport.app is None
        assert transport.runner is None

    @pytest.mark.asyncio
    async def test_error_on_double_start(self) -> None:
        """Test error when starting transport twice."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        await transport.start()

        with pytest.raises(RuntimeError, match="already started"):
            await transport.start()

        await transport.stop()

    @pytest.mark.asyncio
    async def test_create_success_response(self) -> None:
        """Test creating JSON-RPC success response."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = HTTPTransport(server=server)

        response = transport._create_success_response(1, {"result": "ok"})

        assert response.status == 200
        assert response.content_type == "application/json"

    @pytest.mark.asyncio
    async def test_create_error_response(self) -> None:
        """Test creating JSON-RPC error response."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = HTTPTransport(server=server)

        response = transport._create_error_response(
            1, -32600, "Invalid Request", "Bad data"
        )

        assert response.status == 400
        assert response.content_type == "application/json"


class TestSSEConnection:
    """Tests for SSE connection."""

    def test_init(self) -> None:
        """Test SSE connection initialization."""
        response = MagicMock(spec=web.StreamResponse)
        conn = SSEConnection("test-123", response)

        assert conn.connection_id == "test-123"
        assert conn.response == response
        assert conn.connected is True
        assert isinstance(conn.queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_send_event(self) -> None:
        """Test sending SSE event."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock()

        conn = SSEConnection("test-123", response)

        await conn.send_event("test_event", {"data": "test"}, "event-1")

        response.write.assert_called_once()
        call_args = response.write.call_args[0][0]
        assert b"event: test_event" in call_args
        assert b"data:" in call_args

    @pytest.mark.asyncio
    async def test_send_ping(self) -> None:
        """Test sending keep-alive ping."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock()

        conn = SSEConnection("test-123", response)

        await conn.send_ping()

        response.write.assert_called_once_with(b": ping\n\n")

    def test_close(self) -> None:
        """Test closing connection."""
        response = MagicMock(spec=web.StreamResponse)
        conn = SSEConnection("test-123", response)

        assert conn.connected is True

        conn.close()

        assert conn.connected is False


class TestSSETransport:
    """Tests for SSE transport."""

    @pytest.mark.asyncio
    async def test_init(self) -> None:
        """Test SSE transport initialization."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        transport = SSETransport(
            server=server,
            host="localhost",
            port=8080,
            cors_enabled=True,
            cors_origins=["http://localhost:3000"],
            keepalive_interval=15,
        )

        assert transport.server == server
        assert transport.host == "localhost"
        assert transport.port == 8080
        assert transport.cors_enabled is True
        assert transport.cors_origins == ["http://localhost:3000"]
        assert transport.keepalive_interval == 15
        assert transport.app is None
        assert transport.runner is None
        assert len(transport.connections) == 0

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        """Test SSE transport start and stop."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = SSETransport(
            server=server,
            host="localhost",
            port=0,  # Random port
        )

        # Start transport
        await transport.start()
        assert transport.app is not None
        assert transport.runner is not None
        assert transport._keepalive_task is not None

        # Stop transport
        await transport.stop()
        assert transport.app is None
        assert transport.runner is None

    @pytest.mark.asyncio
    async def test_broadcast_event(self) -> None:
        """Test broadcasting events to all connections."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = SSETransport(server=server)

        # Create mock connections
        conn1 = MagicMock(spec=SSEConnection)
        conn1.send_event = AsyncMock()
        conn1.connected = True

        conn2 = MagicMock(spec=SSEConnection)
        conn2.send_event = AsyncMock()
        conn2.connected = True

        transport.connections = {conn1, conn2}

        # Broadcast event
        await transport.broadcast_event("test_event", {"data": "test"})

        conn1.send_event.assert_called_once_with("test_event", {"data": "test"}, None)
        conn2.send_event.assert_called_once_with("test_event", {"data": "test"}, None)


class TestTransportFactory:
    """Tests for transport factory."""

    @pytest.mark.asyncio
    async def test_create_http_transport(self) -> None:
        """Test creating HTTP transport via factory."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        transport = create_transport("http", server, config)

        assert isinstance(transport, HTTPTransport)
        assert transport.server == server
        assert transport.host == config.transport.host
        assert transport.port == config.transport.port

    @pytest.mark.asyncio
    async def test_create_sse_transport(self) -> None:
        """Test creating SSE transport via factory."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        transport = create_transport("sse", server, config)

        assert isinstance(transport, SSETransport)
        assert transport.server == server
        assert transport.host == config.transport.host
        assert transport.port == config.transport.port

    @pytest.mark.asyncio
    async def test_create_invalid_transport(self) -> None:
        """Test error when creating invalid transport type."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        with pytest.raises(ValueError, match="Unsupported transport type"):
            create_transport("invalid", server, config)

    @pytest.mark.asyncio
    async def test_create_transport_case_insensitive(self) -> None:
        """Test factory is case-insensitive."""
        config = get_default_config()
        server = SimplyMCPServer(config)

        transport1 = create_transport("HTTP", server, config)
        transport2 = create_transport("sse", server, config)

        assert isinstance(transport1, HTTPTransport)
        assert isinstance(transport2, SSETransport)
