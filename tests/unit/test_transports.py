"""Unit tests for transport implementations."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web

from simply_mcp.core.config import get_default_config
from simply_mcp.core.server import SimplyMCPServer
from simply_mcp.core.types import PromptConfigModel, ResourceConfigModel, ToolConfigModel
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

        # Check that a rate limiter was created with default values
        assert middleware.rate_limiter is not None
        assert middleware.rate_limiter.requests_per_minute == 60
        assert middleware.rate_limiter.burst_size == 10

    def test_init_custom(self) -> None:
        """Test rate limit middleware with custom values."""
        middleware = RateLimitMiddleware(
            requests_per_minute=120,
            burst_size=20,
        )

        assert middleware.rate_limiter is not None
        assert middleware.rate_limiter.requests_per_minute == 120
        assert middleware.rate_limiter.burst_size == 20

    @pytest.mark.asyncio
    async def test_middleware_tracks_requests(self) -> None:
        """Test rate limit middleware tracks requests."""
        middleware = RateLimitMiddleware(requests_per_minute=600, burst_size=100)

        request = MagicMock(spec=web.Request)
        request.remote = "127.0.0.1"

        response = web.Response()
        handler = AsyncMock(return_value=response)

        # Make a request
        result = await middleware(request, handler)

        assert result == response
        # Check that the rate limiter has tracked the client
        assert "127.0.0.1" in middleware.rate_limiter._clients


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


class TestHTTPTransportAdvanced:
    """Advanced tests for HTTP transport covering uncovered code paths."""

    @pytest.mark.asyncio
    async def test_http_with_rate_limiter(self) -> None:
        """Test HTTP transport with rate limiter."""
        from simply_mcp.security.rate_limiter import RateLimiter

        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        transport = HTTPTransport(
            server=server,
            host="localhost",
            port=0,
            rate_limiter=rate_limiter,
        )

        await transport.start()
        assert transport.app is not None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_http_with_auth_provider(self) -> None:
        """Test HTTP transport with authentication provider."""
        from simply_mcp.security.auth import APIKeyAuthProvider

        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        auth_provider = APIKeyAuthProvider(api_keys=["test-key"])

        transport = HTTPTransport(
            server=server,
            host="localhost",
            port=0,
            auth_provider=auth_provider,
        )

        await transport.start()
        assert transport.app is not None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_handle_root(self) -> None:
        """Test root endpoint handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)
        await transport.start()

        request = MagicMock(spec=web.Request)
        response = await transport.handle_root(request)

        assert response.status == 200
        assert response.content_type == "application/json"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_handle_health(self) -> None:
        """Test health endpoint handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)
        await transport.start()

        request = MagicMock(spec=web.Request)
        response = await transport.handle_health(request)

        assert response.status == 200
        assert response.content_type == "application/json"

        await transport.stop()

    @pytest.mark.asyncio
    async def test_handle_mcp_request_parse_error(self) -> None:
        """Test MCP request handler with JSON parse error."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        # Mock request with invalid JSON
        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid", "", 0))

        response = await transport.handle_mcp_request(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_mcp_request_invalid_structure(self) -> None:
        """Test MCP request handler with invalid request structure."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        # Test with non-dict body
        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(return_value="not a dict")

        response = await transport.handle_mcp_request(request)
        assert response.status == 400

        # Test with missing jsonrpc version
        request.json = AsyncMock(return_value={"method": "test"})
        response = await transport.handle_mcp_request(request)
        assert response.status == 400

        # Test with invalid method type
        request.json = AsyncMock(return_value={"jsonrpc": "2.0", "method": 123})
        response = await transport.handle_mcp_request(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_method_tools_list(self) -> None:
        """Test handling tools/list method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        # Register a tool
        server.registry.register_tool(ToolConfigModel(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "result"
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("tools/list", {})

        assert "tools" in result
        assert len(result["tools"]) > 0

    @pytest.mark.asyncio
    async def test_handle_method_tools_call(self) -> None:
        """Test handling tools/call method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        # Register a tool
        def test_handler(arg: str) -> str:
            return f"Result: {arg}"

        server.registry.register_tool(ToolConfigModel(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
            handler=test_handler
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("tools/call", {"name": "test_tool", "arguments": {"arg": "test"}})

        assert "result" in result
        assert result["result"] == "Result: test"

    @pytest.mark.asyncio
    async def test_handle_method_tools_call_missing_name(self) -> None:
        """Test tools/call with missing tool name."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Tool name is required"):
            await transport._handle_method("tools/call", {})

    @pytest.mark.asyncio
    async def test_handle_method_tools_call_not_found(self) -> None:
        """Test tools/call with non-existent tool."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Tool not found"):
            await transport._handle_method("tools/call", {"name": "nonexistent"})

    @pytest.mark.asyncio
    async def test_handle_method_tools_call_async(self) -> None:
        """Test tools/call with async handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        # Register an async tool
        async def async_handler(arg: str) -> str:
            return f"Async: {arg}"

        server.registry.register_tool(ToolConfigModel(
            name="async_tool",
            description="An async tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
            handler=async_handler
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("tools/call", {"name": "async_tool", "arguments": {"arg": "test"}})

        assert result["result"] == "Async: test"

    @pytest.mark.asyncio
    async def test_handle_method_prompts_list(self) -> None:
        """Test handling prompts/list method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        # Register a prompt
        server.registry.register_prompt(PromptConfigModel(
            name="test_prompt",
            description="A test prompt",
            template="Hello"
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("prompts/list", {})

        assert "prompts" in result
        assert len(result["prompts"]) > 0

    @pytest.mark.asyncio
    async def test_handle_method_prompts_get_with_handler(self) -> None:
        """Test prompts/get with handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        def prompt_handler(name: str) -> str:
            return f"Hello, {name}!"

        server.registry.register_prompt(PromptConfigModel(
            name="test_prompt",
            description="A test prompt",
            handler=prompt_handler
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("prompts/get", {"name": "test_prompt", "arguments": {"name": "World"}})

        assert result["prompt"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_handle_method_prompts_get_with_template(self) -> None:
        """Test prompts/get with template."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        server.registry.register_prompt(PromptConfigModel(
            name="test_prompt",
            description="A test prompt",
            template="Hello, {name}!"
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("prompts/get", {"name": "test_prompt", "arguments": {"name": "World"}})

        assert result["prompt"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_handle_method_prompts_get_missing_name(self) -> None:
        """Test prompts/get with missing prompt name."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Prompt name is required"):
            await transport._handle_method("prompts/get", {})

    @pytest.mark.asyncio
    async def test_handle_method_resources_list(self) -> None:
        """Test handling resources/list method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        # Register a resource
        server.registry.register_resource(ResourceConfigModel(
            uri="file://test.txt",
            name="test",
            description="A test resource",
            mime_type="text/plain",
            handler=lambda: "content"
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("resources/list", {})

        assert "resources" in result

    @pytest.mark.asyncio
    async def test_handle_method_resources_read(self) -> None:
        """Test handling resources/read method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        def resource_handler() -> str:
            return "Resource content"

        server.registry.register_resource(ResourceConfigModel(
            uri="file://test.txt",
            name="test",
            description="A test resource",
            mime_type="text/plain",
            handler=resource_handler
        ))

        transport = HTTPTransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("resources/read", {"uri": "file://test.txt"})

        assert result["content"] == "Resource content"

    @pytest.mark.asyncio
    async def test_handle_method_resources_read_missing_uri(self) -> None:
        """Test resources/read with missing URI."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Resource URI is required"):
            await transport._handle_method("resources/read", {})

    @pytest.mark.asyncio
    async def test_handle_method_unknown(self) -> None:
        """Test handling unknown method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = HTTPTransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Unknown method"):
            await transport._handle_method("unknown/method", {})

    @pytest.mark.asyncio
    async def test_create_error_response_with_data(self) -> None:
        """Test creating error response with additional data."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = HTTPTransport(server=server)

        response = transport._create_error_response(1, -32700, "Parse error", "Invalid JSON")

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_create_error_response_without_data(self) -> None:
        """Test creating error response without additional data."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = HTTPTransport(server=server)

        response = transport._create_error_response(1, -32603, "Internal error")

        assert response.status == 500


class TestSSETransportAdvanced:
    """Advanced tests for SSE transport covering uncovered code paths."""

    @pytest.mark.asyncio
    async def test_sse_with_rate_limiter(self) -> None:
        """Test SSE transport with rate limiter."""
        from simply_mcp.security.rate_limiter import RateLimiter

        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        rate_limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        transport = SSETransport(
            server=server,
            host="localhost",
            port=0,
            rate_limiter=rate_limiter,
        )

        await transport.start()
        assert transport.app is not None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sse_with_auth_provider(self) -> None:
        """Test SSE transport with authentication provider."""
        from simply_mcp.security.auth import APIKeyAuthProvider

        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        auth_provider = APIKeyAuthProvider(api_keys=["test-key"])

        transport = SSETransport(
            server=server,
            host="localhost",
            port=0,
            auth_provider=auth_provider,
        )

        await transport.start()
        assert transport.app is not None

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sse_handle_root(self) -> None:
        """Test SSE root endpoint handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = SSETransport(server=server, host="localhost", port=0)
        await transport.start()

        request = MagicMock(spec=web.Request)
        response = await transport.handle_root(request)

        assert response.status == 200

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sse_handle_health(self) -> None:
        """Test SSE health endpoint handler."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = SSETransport(server=server, host="localhost", port=0)
        await transport.start()

        request = MagicMock(spec=web.Request)
        response = await transport.handle_health(request)

        assert response.status == 200

        await transport.stop()

    @pytest.mark.asyncio
    async def test_sse_handle_mcp_request_parse_error(self) -> None:
        """Test SSE MCP request handler with JSON parse error."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = SSETransport(server=server, host="localhost", port=0)

        # Mock request with invalid JSON
        request = MagicMock(spec=web.Request)
        request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid", "", 0))

        response = await transport.handle_mcp_request(request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_sse_handle_method_tools_list(self) -> None:
        """Test SSE handling tools/list method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        server.registry.register_tool(ToolConfigModel(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "result"
        ))

        transport = SSETransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("tools/list", {})

        assert "tools" in result

    @pytest.mark.asyncio
    async def test_sse_handle_method_tools_call(self) -> None:
        """Test SSE handling tools/call method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        def test_handler(arg: str) -> str:
            return f"Result: {arg}"

        server.registry.register_tool(ToolConfigModel(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg": {"type": "string"}}},
            handler=test_handler
        ))

        transport = SSETransport(server=server, host="localhost", port=0)

        result = await transport._handle_method("tools/call", {"name": "test_tool", "arguments": {"arg": "test"}})

        assert result["result"] == "Result: test"

    @pytest.mark.asyncio
    async def test_sse_handle_method_unknown(self) -> None:
        """Test SSE handling unknown method."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        await server.initialize()

        transport = SSETransport(server=server, host="localhost", port=0)

        with pytest.raises(ValueError, match="Unknown method"):
            await transport._handle_method("unknown/method", {})

    @pytest.mark.asyncio
    async def test_sse_broadcast_no_connections(self) -> None:
        """Test broadcasting with no connections."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = SSETransport(server=server)

        # Should not raise error
        await transport.broadcast_event("test", {"data": "test"})

    @pytest.mark.asyncio
    async def test_sse_broadcast_error_handling(self) -> None:
        """Test broadcast error handling."""
        config = get_default_config()
        server = SimplyMCPServer(config)
        transport = SSETransport(server=server)

        # Create mock connection that raises error
        conn = MagicMock(spec=SSEConnection)
        conn.send_event = AsyncMock(side_effect=Exception("Send failed"))
        conn.connection_id = "test-123"

        transport.connections = {conn}

        # Should not raise error
        await transport.broadcast_event("test", {"data": "test"})

    @pytest.mark.asyncio
    async def test_sse_connection_send_event_when_disconnected(self) -> None:
        """Test sending event when connection is disconnected."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock()

        conn = SSEConnection("test-123", response)
        conn.connected = False

        # Should not call write
        await conn.send_event("test", {"data": "test"})

        response.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_sse_connection_send_event_error(self) -> None:
        """Test send event error handling."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock(side_effect=Exception("Write failed"))

        conn = SSEConnection("test-123", response)

        # Should set connected to False
        await conn.send_event("test", {"data": "test"})

        assert conn.connected is False

    @pytest.mark.asyncio
    async def test_sse_connection_send_ping_when_disconnected(self) -> None:
        """Test sending ping when connection is disconnected."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock()

        conn = SSEConnection("test-123", response)
        conn.connected = False

        # Should not call write
        await conn.send_ping()

        response.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_sse_connection_send_ping_error(self) -> None:
        """Test send ping error handling."""
        response = MagicMock(spec=web.StreamResponse)
        response.write = AsyncMock(side_effect=Exception("Write failed"))

        conn = SSEConnection("test-123", response)

        # Should set connected to False
        await conn.send_ping()

        assert conn.connected is False


import json
