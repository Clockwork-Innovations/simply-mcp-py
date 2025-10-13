"""Unit tests for the SimplyMCPServer class.

This module tests the core MCP server implementation, including:
- Server initialization
- Component registration (tools, prompts, resources)
- Handler execution
- MCP protocol integration
- Error handling
- Lifecycle management
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import mcp.types as types
import pytest
from pydantic import AnyUrl

from simply_mcp.core.config import get_default_config
from simply_mcp.core.errors import (
    HandlerExecutionError,
    HandlerNotFoundError,
    ValidationError,
)
from simply_mcp.core.server import SimplyMCPServer
from simply_mcp.core.types import PromptConfig, ResourceConfig, ToolConfig


class TestSimplyMCPServerInit:
    """Test server initialization."""

    def test_init_with_default_config(self) -> None:
        """Test server initialization with default config."""
        server = SimplyMCPServer()

        assert server.config is not None
        assert server.registry is not None
        assert server.mcp_server is not None
        assert not server.is_initialized
        assert not server.is_running
        assert server.request_count == 0

    def test_init_with_custom_config(self) -> None:
        """Test server initialization with custom config."""
        config = get_default_config()
        config.server.name = "custom-server"
        config.server.version = "1.2.3"

        server = SimplyMCPServer(config)

        assert server.config.server.name == "custom-server"
        assert server.config.server.version == "1.2.3"
        assert server.mcp_server.name == "custom-server"
        assert server.mcp_server.version == "1.2.3"

    def test_mcp_server_created_with_metadata(self) -> None:
        """Test MCP server is created with correct metadata."""
        config = get_default_config()
        config.server.name = "test-server"
        config.server.version = "0.1.0"
        config.server.description = "Test description"
        config.server.homepage = "https://example.com"

        server = SimplyMCPServer(config)

        assert server.mcp_server.name == "test-server"
        assert server.mcp_server.version == "0.1.0"
        assert server.mcp_server.instructions == "Test description"
        assert server.mcp_server.website_url == "https://example.com"


class TestSimplyMCPServerInitialize:
    """Test server initialization."""

    @pytest.mark.asyncio
    async def test_initialize_success(self) -> None:
        """Test successful server initialization."""
        server = SimplyMCPServer()

        await server.initialize()

        assert server.is_initialized
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_initialize_twice_raises_error(self) -> None:
        """Test that initializing twice raises an error."""
        server = SimplyMCPServer()
        await server.initialize()

        with pytest.raises(RuntimeError, match="already initialized"):
            await server.initialize()

    @pytest.mark.asyncio
    async def test_initialize_registers_handlers(self) -> None:
        """Test that initialization registers MCP handlers."""
        server = SimplyMCPServer()

        await server.initialize()

        # Check that handlers are registered in MCP server
        mcp_server = server.get_mcp_server()
        assert types.ListToolsRequest in mcp_server.request_handlers
        assert types.CallToolRequest in mcp_server.request_handlers
        assert types.ListPromptsRequest in mcp_server.request_handlers
        assert types.GetPromptRequest in mcp_server.request_handlers
        assert types.ListResourcesRequest in mcp_server.request_handlers
        assert types.ReadResourceRequest in mcp_server.request_handlers


class TestSimplyMCPServerToolRegistration:
    """Test tool registration."""

    def test_register_tool_success(self) -> None:
        """Test successful tool registration."""
        server = SimplyMCPServer()

        def add(a: int, b: int) -> int:
            return a + b

        tool_config: ToolConfig = {
            "name": "add",
            "description": "Add two numbers",
            "input_schema": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            "handler": add,
        }

        server.register_tool(tool_config)

        # Check tool is in registry
        assert server.registry.has_tool("add")
        retrieved_tool = server.registry.get_tool("add")
        assert retrieved_tool is not None
        assert retrieved_tool["name"] == "add"
        assert retrieved_tool["handler"] == add

    def test_register_duplicate_tool_raises_error(self) -> None:
        """Test that registering duplicate tool raises error."""
        server = SimplyMCPServer()

        def add(a: int, b: int) -> int:
            return a + b

        tool_config: ToolConfig = {
            "name": "add",
            "description": "Add two numbers",
            "input_schema": {"type": "object"},
            "handler": add,
        }

        server.register_tool(tool_config)

        with pytest.raises(ValidationError, match="already registered"):
            server.register_tool(tool_config)

    def test_register_multiple_tools(self) -> None:
        """Test registering multiple tools."""
        server = SimplyMCPServer()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        server.register_tool(
            {
                "name": "add",
                "description": "Add",
                "input_schema": {"type": "object"},
                "handler": add,
            }
        )
        server.register_tool(
            {
                "name": "multiply",
                "description": "Multiply",
                "input_schema": {"type": "object"},
                "handler": multiply,
            }
        )

        assert server.registry.has_tool("add")
        assert server.registry.has_tool("multiply")


class TestSimplyMCPServerPromptRegistration:
    """Test prompt registration."""

    def test_register_prompt_with_template(self) -> None:
        """Test registering prompt with static template."""
        server = SimplyMCPServer()

        prompt_config: PromptConfig = {
            "name": "greeting",
            "description": "Generate a greeting",
            "template": "Hello, {name}!",
            "arguments": ["name"],
        }

        server.register_prompt(prompt_config)

        assert server.registry.has_prompt("greeting")
        retrieved = server.registry.get_prompt("greeting")
        assert retrieved is not None
        assert retrieved["name"] == "greeting"
        assert retrieved["template"] == "Hello, {name}!"

    def test_register_prompt_with_handler(self) -> None:
        """Test registering prompt with dynamic handler."""
        server = SimplyMCPServer()

        def generate_greeting(name: str) -> str:
            return f"Hello, {name}!"

        prompt_config: PromptConfig = {
            "name": "greeting",
            "description": "Generate a greeting",
            "handler": generate_greeting,
            "arguments": ["name"],
        }

        server.register_prompt(prompt_config)

        assert server.registry.has_prompt("greeting")

    def test_register_duplicate_prompt_raises_error(self) -> None:
        """Test that registering duplicate prompt raises error."""
        server = SimplyMCPServer()

        prompt_config: PromptConfig = {
            "name": "greeting",
            "description": "Generate a greeting",
            "template": "Hello!",
        }

        server.register_prompt(prompt_config)

        with pytest.raises(ValidationError, match="already registered"):
            server.register_prompt(prompt_config)


class TestSimplyMCPServerResourceRegistration:
    """Test resource registration."""

    def test_register_resource_success(self) -> None:
        """Test successful resource registration."""
        server = SimplyMCPServer()

        def load_config() -> Dict[str, Any]:
            return {"key": "value"}

        resource_config: ResourceConfig = {
            "uri": "config://app",
            "name": "config",
            "description": "App configuration",
            "mime_type": "application/json",
            "handler": load_config,
        }

        server.register_resource(resource_config)

        assert server.registry.has_resource("config://app")

    def test_register_duplicate_resource_raises_error(self) -> None:
        """Test that registering duplicate resource raises error."""
        server = SimplyMCPServer()

        def load_config() -> Dict[str, Any]:
            return {"key": "value"}

        resource_config: ResourceConfig = {
            "uri": "config://app",
            "name": "config",
            "description": "App configuration",
            "mime_type": "application/json",
            "handler": load_config,
        }

        server.register_resource(resource_config)

        with pytest.raises(ValidationError, match="already registered"):
            server.register_resource(resource_config)


class TestSimplyMCPServerListTools:
    """Test list_tools handler."""

    @pytest.mark.asyncio
    async def test_list_tools_empty(self) -> None:
        """Test listing tools when none are registered."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListToolsRequest]

        # Call handler (SDK will accept None)
        request = types.ListToolsRequest(method="tools/list")
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert hasattr(result.root, "tools")
        assert len(result.root.tools) == 0

    @pytest.mark.asyncio
    async def test_list_tools_with_registered_tools(self) -> None:
        """Test listing tools with registered tools."""
        server = SimplyMCPServer()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        server.register_tool(
            {
                "name": "add",
                "description": "Add two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
                "handler": add,
            }
        )
        server.register_tool(
            {
                "name": "multiply",
                "description": "Multiply two numbers",
                "input_schema": {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                },
                "handler": multiply,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListToolsRequest]

        # Call handler
        request = types.ListToolsRequest(method="tools/list")
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert hasattr(result.root, "tools")
        assert len(result.root.tools) == 2

        tool_names = [tool.name for tool in result.root.tools]
        assert "add" in tool_names
        assert "multiply" in tool_names


class TestSimplyMCPServerCallTool:
    """Test call_tool handler."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self) -> None:
        """Test successful tool execution."""
        server = SimplyMCPServer()

        def add(a: int, b: int) -> int:
            return a + b

        server.register_tool(
            {
                "name": "add",
                "description": "Add two numbers",
                "input_schema": {"type": "object"},
                "handler": add,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.CallToolRequest]

        # Call handler
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="add", arguments={"a": 5, "b": 3}),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert hasattr(result.root, "content")
        assert len(result.root.content) > 0
        assert "8" in result.root.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self) -> None:
        """Test calling non-existent tool."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.CallToolRequest]

        # Call handler with non-existent tool
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="nonexistent", arguments={}),
        )

        # MCP SDK's call_tool decorator catches exceptions and returns error result
        result = await handler(request)
        assert isinstance(result, types.ServerResult)
        assert result.root.isError is True

    @pytest.mark.asyncio
    async def test_call_tool_execution_error(self) -> None:
        """Test tool execution error handling."""
        server = SimplyMCPServer()

        def failing_tool() -> int:
            raise ValueError("Intentional error")

        server.register_tool(
            {
                "name": "failing",
                "description": "A tool that fails",
                "input_schema": {"type": "object"},
                "handler": failing_tool,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.CallToolRequest]

        # Call handler
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="failing", arguments={}),
        )

        # MCP SDK's call_tool decorator catches exceptions and returns error result
        result = await handler(request)
        assert isinstance(result, types.ServerResult)
        assert result.root.isError is True
        assert "Intentional error" in result.root.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_async_handler(self) -> None:
        """Test tool with async handler."""
        server = SimplyMCPServer()

        async def async_add(a: int, b: int) -> int:
            await asyncio.sleep(0.01)  # Simulate async work
            return a + b

        server.register_tool(
            {
                "name": "async_add",
                "description": "Async add",
                "input_schema": {"type": "object"},
                "handler": async_add,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.CallToolRequest]

        # Call handler
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name="async_add", arguments={"a": 10, "b": 5}
            ),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert "15" in result.root.content[0].text

    @pytest.mark.asyncio
    async def test_call_tool_dict_result(self) -> None:
        """Test tool returning dictionary."""
        server = SimplyMCPServer()

        def get_user() -> Dict[str, Any]:
            return {"name": "John", "age": 30}

        server.register_tool(
            {
                "name": "get_user",
                "description": "Get user info",
                "input_schema": {"type": "object"},
                "handler": get_user,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.CallToolRequest]

        # Call handler
        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name="get_user", arguments={}),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert "John" in result.root.content[0].text
        assert "30" in result.root.content[0].text


class TestSimplyMCPServerListPrompts:
    """Test list_prompts handler."""

    @pytest.mark.asyncio
    async def test_list_prompts_empty(self) -> None:
        """Test listing prompts when none are registered."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListPromptsRequest]

        # Call handler
        request = types.ListPromptsRequest(
            method="prompts/list", params=types.ListPromptsRequestParams()
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert hasattr(result.root, "prompts")
        assert len(result.root.prompts) == 0

    @pytest.mark.asyncio
    async def test_list_prompts_with_registered_prompts(self) -> None:
        """Test listing prompts with registered prompts."""
        server = SimplyMCPServer()

        server.register_prompt(
            {
                "name": "greeting",
                "description": "Generate greeting",
                "template": "Hello!",
            }
        )
        server.register_prompt(
            {
                "name": "farewell",
                "description": "Generate farewell",
                "template": "Goodbye!",
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListPromptsRequest]

        # Call handler
        request = types.ListPromptsRequest(
            method="prompts/list", params=types.ListPromptsRequestParams()
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert len(result.root.prompts) == 2

        prompt_names = [p.name for p in result.root.prompts]
        assert "greeting" in prompt_names
        assert "farewell" in prompt_names


class TestSimplyMCPServerGetPrompt:
    """Test get_prompt handler."""

    @pytest.mark.asyncio
    async def test_get_prompt_with_template(self) -> None:
        """Test getting prompt with static template."""
        server = SimplyMCPServer()

        server.register_prompt(
            {
                "name": "greeting",
                "description": "Generate greeting",
                "template": "Hello, {name}!",
                "arguments": ["name"],
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.GetPromptRequest]

        # Call handler
        request = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="greeting", arguments={"name": "Alice"}),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert isinstance(result.root, types.GetPromptResult)
        assert len(result.root.messages) > 0
        assert "Hello, Alice!" in result.root.messages[0].content.text

    @pytest.mark.asyncio
    async def test_get_prompt_with_handler(self) -> None:
        """Test getting prompt with dynamic handler."""
        server = SimplyMCPServer()

        def generate_greeting(name: str) -> str:
            return f"Greetings, {name}!"

        server.register_prompt(
            {
                "name": "greeting",
                "description": "Generate greeting",
                "handler": generate_greeting,
                "arguments": ["name"],
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.GetPromptRequest]

        # Call handler
        request = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="greeting", arguments={"name": "Bob"}),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert "Greetings, Bob!" in result.root.messages[0].content.text

    @pytest.mark.asyncio
    async def test_get_prompt_not_found(self) -> None:
        """Test getting non-existent prompt."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.GetPromptRequest]

        # Call handler
        request = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="nonexistent", arguments=None),
        )

        with pytest.raises(HandlerNotFoundError):
            await handler(request)

    @pytest.mark.asyncio
    async def test_get_prompt_async_handler(self) -> None:
        """Test prompt with async handler."""
        server = SimplyMCPServer()

        async def async_greeting(name: str) -> str:
            await asyncio.sleep(0.01)
            return f"Async hello, {name}!"

        server.register_prompt(
            {
                "name": "greeting",
                "description": "Generate greeting",
                "handler": async_greeting,
                "arguments": ["name"],
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.GetPromptRequest]

        # Call handler
        request = types.GetPromptRequest(
            method="prompts/get",
            params=types.GetPromptRequestParams(name="greeting", arguments={"name": "Charlie"}),
        )
        result = await handler(request)

        # Check result
        assert "Async hello, Charlie!" in result.root.messages[0].content.text


class TestSimplyMCPServerListResources:
    """Test list_resources handler."""

    @pytest.mark.asyncio
    async def test_list_resources_empty(self) -> None:
        """Test listing resources when none are registered."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListResourcesRequest]

        # Call handler
        request = types.ListResourcesRequest(
            method="resources/list", params=types.ListResourcesRequestParams()
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert hasattr(result.root, "resources")
        assert len(result.root.resources) == 0

    @pytest.mark.asyncio
    async def test_list_resources_with_registered_resources(self) -> None:
        """Test listing resources with registered resources."""
        server = SimplyMCPServer()

        def load_config() -> Dict[str, Any]:
            return {"key": "value"}

        def load_schema() -> Dict[str, Any]:
            return {"type": "object"}

        server.register_resource(
            {
                "uri": "config://app",
                "name": "config",
                "description": "App config",
                "mime_type": "application/json",
                "handler": load_config,
            }
        )
        server.register_resource(
            {
                "uri": "schema://app",
                "name": "schema",
                "description": "App schema",
                "mime_type": "application/json",
                "handler": load_schema,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ListResourcesRequest]

        # Call handler
        request = types.ListResourcesRequest(
            method="resources/list", params=types.ListResourcesRequestParams()
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, types.ServerResult)
        assert len(result.root.resources) == 2

        resource_uris = [str(r.uri) for r in result.root.resources]
        assert "config://app" in resource_uris
        assert "schema://app" in resource_uris


class TestSimplyMCPServerReadResource:
    """Test read_resource handler."""

    @pytest.mark.asyncio
    async def test_read_resource_success(self) -> None:
        """Test successful resource reading."""
        server = SimplyMCPServer()

        def load_config() -> Dict[str, Any]:
            return {"key": "value"}

        server.register_resource(
            {
                "uri": "config://app",
                "name": "config",
                "description": "App config",
                "mime_type": "application/json",
                "handler": load_config,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ReadResourceRequest]

        # Call handler
        request = types.ReadResourceRequest(
            method="resources/read",
            params=types.ReadResourceRequestParams(uri=AnyUrl("config://app")),
        )
        result = await handler(request)

        # Check result
        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    @pytest.mark.asyncio
    async def test_read_resource_not_found(self) -> None:
        """Test reading non-existent resource."""
        server = SimplyMCPServer()
        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ReadResourceRequest]

        # Call handler
        request = types.ReadResourceRequest(
            method="resources/read",
            params=types.ReadResourceRequestParams(uri=AnyUrl("config://nonexistent")),
        )

        with pytest.raises(HandlerNotFoundError):
            await handler(request)

    @pytest.mark.asyncio
    async def test_read_resource_string_result(self) -> None:
        """Test resource returning string."""
        server = SimplyMCPServer()

        def load_text() -> str:
            return "Hello, world!"

        server.register_resource(
            {
                "uri": "text://greeting",
                "name": "greeting",
                "description": "Greeting text",
                "mime_type": "text/plain",
                "handler": load_text,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ReadResourceRequest]

        # Call handler
        request = types.ReadResourceRequest(
            method="resources/read",
            params=types.ReadResourceRequestParams(uri=AnyUrl("text://greeting")),
        )
        result = await handler(request)

        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_read_resource_async_handler(self) -> None:
        """Test resource with async handler."""
        server = SimplyMCPServer()

        async def async_load() -> str:
            await asyncio.sleep(0.01)
            return "Async content"

        server.register_resource(
            {
                "uri": "async://content",
                "name": "content",
                "description": "Async content",
                "mime_type": "text/plain",
                "handler": async_load,
            }
        )

        await server.initialize()

        # Get the handler
        mcp_server = server.get_mcp_server()
        handler = mcp_server.request_handlers[types.ReadResourceRequest]

        # Call handler
        request = types.ReadResourceRequest(
            method="resources/read",
            params=types.ReadResourceRequestParams(uri=AnyUrl("async://content")),
        )
        result = await handler(request)

        assert result == "Async content"


class TestSimplyMCPServerLifecycle:
    """Test server lifecycle methods."""

    @pytest.mark.asyncio
    async def test_run_stdio_not_initialized_raises_error(self) -> None:
        """Test that run_stdio raises error if not initialized."""
        server = SimplyMCPServer()

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.run_stdio()

    @pytest.mark.asyncio
    async def test_run_with_transport_not_initialized_raises_error(self) -> None:
        """Test that run_with_transport raises error if not initialized."""
        server = SimplyMCPServer()

        read_stream = Mock()
        write_stream = Mock()

        with pytest.raises(RuntimeError, match="not initialized"):
            await server.run_with_transport(read_stream, write_stream)

    @pytest.mark.asyncio
    async def test_stop_server(self) -> None:
        """Test stopping the server."""
        server = SimplyMCPServer()
        await server.initialize()

        # Server not running yet
        assert not server.is_running

        # Stop should succeed
        await server.stop()
        assert not server.is_running

    @pytest.mark.asyncio
    async def test_shutdown_server(self) -> None:
        """Test shutting down the server."""
        server = SimplyMCPServer()
        await server.initialize()

        assert server.is_initialized

        await server.shutdown()

        assert not server.is_initialized
        assert not server.is_running


class TestSimplyMCPServerProperties:
    """Test server properties."""

    def test_get_mcp_server(self) -> None:
        """Test getting underlying MCP server."""
        server = SimplyMCPServer()

        mcp_server = server.get_mcp_server()

        assert mcp_server is not None
        assert mcp_server == server.mcp_server

    def test_is_initialized_property(self) -> None:
        """Test is_initialized property."""
        server = SimplyMCPServer()

        assert not server.is_initialized

    @pytest.mark.asyncio
    async def test_is_initialized_after_init(self) -> None:
        """Test is_initialized after initialization."""
        server = SimplyMCPServer()

        await server.initialize()

        assert server.is_initialized

    def test_is_running_property(self) -> None:
        """Test is_running property."""
        server = SimplyMCPServer()

        assert not server.is_running

    def test_request_count_property(self) -> None:
        """Test request_count property."""
        server = SimplyMCPServer()

        assert server.request_count == 0


class TestSimplyMCPServerIntegration:
    """Integration tests for server."""

    @pytest.mark.asyncio
    async def test_full_tool_workflow(self) -> None:
        """Test complete workflow: register, list, call tool."""
        server = SimplyMCPServer()

        # Register tool
        def calculate(x: int, y: int, operation: str) -> int:
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            else:
                raise ValueError(f"Unknown operation: {operation}")

        server.register_tool(
            {
                "name": "calculate",
                "description": "Perform calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "operation": {"type": "string"},
                    },
                    "required": ["x", "y", "operation"],
                },
                "handler": calculate,
            }
        )

        await server.initialize()

        mcp_server = server.get_mcp_server()

        # List tools
        list_handler = mcp_server.request_handlers[types.ListToolsRequest]
        list_result = await list_handler(
            types.ListToolsRequest(
                method="tools/list", params=types.ListToolsRequestParams()
            )
        )
        assert len(list_result.root.tools) == 1
        assert list_result.root.tools[0].name == "calculate"

        # Call tool - add
        call_handler = mcp_server.request_handlers[types.CallToolRequest]
        call_result = await call_handler(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="calculate", arguments={"x": 10, "y": 5, "operation": "add"}
                ),
            )
        )
        assert "15" in call_result.root.content[0].text

        # Call tool - multiply
        call_result = await call_handler(
            types.CallToolRequest(
                method="tools/call",
                params=types.CallToolRequestParams(
                    name="calculate", arguments={"x": 10, "y": 5, "operation": "multiply"}
                ),
            )
        )
        assert "50" in call_result.root.content[0].text

        # Verify request count
        assert server.request_count == 2

    @pytest.mark.asyncio
    async def test_full_prompt_workflow(self) -> None:
        """Test complete workflow: register, list, get prompt."""
        server = SimplyMCPServer()

        # Register prompt
        server.register_prompt(
            {
                "name": "summarize",
                "description": "Generate a summary prompt",
                "template": "Please summarize the following text: {text}",
                "arguments": ["text"],
            }
        )

        await server.initialize()

        mcp_server = server.get_mcp_server()

        # List prompts
        list_handler = mcp_server.request_handlers[types.ListPromptsRequest]
        list_result = await list_handler(
            types.ListPromptsRequest(
                method="prompts/list", params=types.ListPromptsRequestParams()
            )
        )
        assert len(list_result.root.prompts) == 1
        assert list_result.root.prompts[0].name == "summarize"

        # Get prompt
        get_handler = mcp_server.request_handlers[types.GetPromptRequest]
        get_result = await get_handler(
            types.GetPromptRequest(
                method="prompts/get",
                params=types.GetPromptRequestParams(
                    name="summarize", arguments={"text": "Lorem ipsum dolor sit amet"}
                ),
            )
        )
        assert "Lorem ipsum dolor sit amet" in get_result.root.messages[0].content.text

        # Verify request count
        assert server.request_count == 1

    @pytest.mark.asyncio
    async def test_full_resource_workflow(self) -> None:
        """Test complete workflow: register, list, read resource."""
        server = SimplyMCPServer()

        # Register resource
        def load_data() -> Dict[str, Any]:
            return {"users": ["alice", "bob"], "count": 2}

        server.register_resource(
            {
                "uri": "data://users",
                "name": "users",
                "description": "User data",
                "mime_type": "application/json",
                "handler": load_data,
            }
        )

        await server.initialize()

        mcp_server = server.get_mcp_server()

        # List resources
        list_handler = mcp_server.request_handlers[types.ListResourcesRequest]
        list_result = await list_handler(
            types.ListResourcesRequest(
                method="resources/list", params=types.ListResourcesRequestParams()
            )
        )
        assert len(list_result.root.resources) == 1
        assert str(list_result.root.resources[0].uri) == "data://users"

        # Read resource
        read_handler = mcp_server.request_handlers[types.ReadResourceRequest]
        read_result = await read_handler(
            types.ReadResourceRequest(
                method="resources/read",
                params=types.ReadResourceRequestParams(uri=AnyUrl("data://users")),
            )
        )
        assert "alice" in read_result
        assert "bob" in read_result

        # Verify request count
        assert server.request_count == 1
