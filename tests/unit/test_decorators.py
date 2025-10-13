"""Unit tests for decorator API.

Tests the @tool, @prompt, @resource, and @mcp_server decorators, including:
- Auto-schema generation
- Pydantic model support
- Method decorators (class-based)
- Global server management
- Metadata storage
- Error cases
"""

import pytest
from typing import Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from simply_mcp.api.decorators import (
    get_global_server,
    mcp_server,
    prompt,
    resource,
    set_global_server,
    tool,
)
from simply_mcp.core.config import SimplyMCPConfig, ServerMetadataModel
from simply_mcp.core.server import SimplyMCPServer


@pytest.fixture
def reset_global_server():
    """Reset global server before each test."""
    # Import to access the module-level variable
    import simply_mcp.api.decorators as decorators_module

    # Store original
    original = decorators_module._global_server

    # Reset to None
    decorators_module._global_server = None

    yield

    # Restore original
    decorators_module._global_server = original


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_tool_decorator_auto_schema(self, reset_global_server):
        """Test @tool with auto-generated schema."""
        @tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        # Check metadata
        assert hasattr(add, '_mcp_tool_config')
        assert hasattr(add, '_mcp_component_type')
        assert add._mcp_component_type == 'tool'

        # Check config
        config = add._mcp_tool_config
        assert config['name'] == 'add'
        assert config['description'] == 'Add two numbers.'
        assert 'input_schema' in config
        assert config['input_schema']['type'] == 'object'
        assert 'a' in config['input_schema']['properties']
        assert 'b' in config['input_schema']['properties']
        assert config['input_schema']['properties']['a']['type'] == 'integer'
        assert config['input_schema']['properties']['b']['type'] == 'integer'
        assert config['handler'] == add

        # Check registered with global server
        server = get_global_server()
        registered_tool = server.registry.get_tool('add')
        assert registered_tool is not None
        assert registered_tool['name'] == 'add'

    def test_tool_decorator_custom_name_description(self, reset_global_server):
        """Test @tool with custom name and description."""
        @tool(name="custom_add", description="Custom addition tool")
        def add(a: int, b: int) -> int:
            return a + b

        config = add._mcp_tool_config
        assert config['name'] == 'custom_add'
        assert config['description'] == 'Custom addition tool'

        # Check registered with custom name
        server = get_global_server()
        registered_tool = server.registry.get_tool('custom_add')
        assert registered_tool is not None

    def test_tool_decorator_explicit_schema(self, reset_global_server):
        """Test @tool with explicit schema."""
        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer"}
            },
            "required": ["x"]
        }

        @tool(input_schema=schema)
        def process(x: int) -> int:
            """Process a number."""
            return x * 2

        config = process._mcp_tool_config
        assert config['input_schema'] == schema

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_tool_decorator_pydantic_model(self, reset_global_server):
        """Test @tool with Pydantic model."""
        class SearchInput(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, ge=1, le=100)

        @tool(input_schema=SearchInput)
        def search(input: SearchInput) -> list:
            """Search with validation."""
            return [f"Result for {input.query}"]

        config = search._mcp_tool_config
        assert 'input_schema' in config
        schema = config['input_schema']
        assert schema['type'] == 'object'
        assert 'query' in schema['properties']
        assert 'limit' in schema['properties']
        assert schema['properties']['limit']['default'] == 10

    def test_tool_decorator_preserves_function_metadata(self, reset_global_server):
        """Test that @tool preserves function metadata."""
        @tool()
        def my_function(x: int) -> int:
            """My docstring."""
            return x

        assert my_function.__name__ == 'my_function'
        assert my_function.__doc__ == 'My docstring.'

    def test_tool_decorator_without_docstring(self, reset_global_server):
        """Test @tool on function without docstring."""
        @tool()
        def no_doc(x: int) -> int:
            return x * 2

        config = no_doc._mcp_tool_config
        assert config['description'] == 'Tool: no_doc'

    def test_tool_decorator_optional_parameters(self, reset_global_server):
        """Test @tool with optional parameters."""
        @tool()
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        config = greet._mcp_tool_config
        schema = config['input_schema']
        assert 'name' in schema['required']
        assert 'greeting' not in schema['required']
        assert schema['properties']['greeting']['default'] == "Hello"


class TestPromptDecorator:
    """Tests for @prompt decorator."""

    def test_prompt_decorator_auto_detection(self, reset_global_server):
        """Test @prompt with auto-detected arguments."""
        @prompt()
        def code_review(language: str = "python") -> str:
            """Generate a code review prompt."""
            return f"Review this {language} code..."

        # Check metadata
        assert hasattr(code_review, '_mcp_prompt_config')
        assert hasattr(code_review, '_mcp_component_type')
        assert code_review._mcp_component_type == 'prompt'

        # Check config
        config = code_review._mcp_prompt_config
        assert config['name'] == 'code_review'
        assert config['description'] == 'Generate a code review prompt.'
        assert config['arguments'] == ['language']
        assert config['handler'] == code_review

        # Check registered with global server
        server = get_global_server()
        registered_prompt = server.registry.get_prompt('code_review')
        assert registered_prompt is not None

    def test_prompt_decorator_custom_name_description(self, reset_global_server):
        """Test @prompt with custom name and description."""
        @prompt(name="custom_prompt", description="Custom description")
        def my_prompt(topic: str) -> str:
            return f"Prompt about {topic}"

        config = my_prompt._mcp_prompt_config
        assert config['name'] == 'custom_prompt'
        assert config['description'] == 'Custom description'

    def test_prompt_decorator_explicit_arguments(self, reset_global_server):
        """Test @prompt with explicit arguments."""
        @prompt(arguments=["topic", "style"])
        def generate_prompt(topic: str, style: str = "formal") -> str:
            """Generate a writing prompt."""
            return f"Write about {topic} in a {style} style"

        config = generate_prompt._mcp_prompt_config
        assert config['arguments'] == ["topic", "style"]

    def test_prompt_decorator_no_arguments(self, reset_global_server):
        """Test @prompt with no arguments."""
        @prompt()
        def static_prompt() -> str:
            """Static prompt."""
            return "This is a static prompt"

        config = static_prompt._mcp_prompt_config
        # When there are no arguments, the arguments key is not present
        # This is correct behavior - PromptConfig has 'arguments' as NotRequired
        assert 'arguments' not in config or config.get('arguments') == []

    def test_prompt_decorator_without_docstring(self, reset_global_server):
        """Test @prompt without docstring."""
        @prompt()
        def no_doc() -> str:
            return "Prompt text"

        config = no_doc._mcp_prompt_config
        assert config['description'] == 'Prompt: no_doc'


class TestResourceDecorator:
    """Tests for @resource decorator."""

    def test_resource_decorator_basic(self, reset_global_server):
        """Test @resource with basic configuration."""
        @resource(uri="config://app")
        def get_config() -> dict:
            """Get application config."""
            return {"version": "1.0.0"}

        # Check metadata
        assert hasattr(get_config, '_mcp_resource_config')
        assert hasattr(get_config, '_mcp_component_type')
        assert get_config._mcp_component_type == 'resource'

        # Check config
        config = get_config._mcp_resource_config
        assert config['uri'] == 'config://app'
        assert config['name'] == 'get_config'
        assert config['description'] == 'Get application config.'
        assert config['mime_type'] == 'application/json'
        assert config['handler'] == get_config

        # Check registered with global server
        server = get_global_server()
        registered_resource = server.registry.get_resource('config://app')
        assert registered_resource is not None

    def test_resource_decorator_custom_name_description(self, reset_global_server):
        """Test @resource with custom name and description."""
        @resource(
            uri="data://stats",
            name="statistics",
            description="System statistics"
        )
        def get_stats() -> str:
            return "CPU: 50%"

        config = get_stats._mcp_resource_config
        assert config['name'] == 'statistics'
        assert config['description'] == 'System statistics'

    def test_resource_decorator_custom_mime_type(self, reset_global_server):
        """Test @resource with custom MIME type."""
        @resource(uri="data://text", mime_type="text/plain")
        def get_text() -> str:
            """Get text data."""
            return "Plain text content"

        config = get_text._mcp_resource_config
        assert config['mime_type'] == 'text/plain'

    def test_resource_decorator_missing_uri(self, reset_global_server):
        """Test @resource raises error when URI is missing."""
        with pytest.raises(ValueError, match="Resource URI is required"):
            @resource(uri="")
            def bad_resource() -> dict:
                return {}

    def test_resource_decorator_without_docstring(self, reset_global_server):
        """Test @resource without docstring."""
        @resource(uri="test://resource")
        def no_doc() -> dict:
            return {}

        config = no_doc._mcp_resource_config
        assert config['description'] == 'Resource: no_doc'


class TestMCPServerDecorator:
    """Tests for @mcp_server class decorator."""

    def test_mcp_server_decorator_basic(self, reset_global_server):
        """Test @mcp_server with basic class."""
        @mcp_server(name="calculator", version="1.0.0")
        class Calculator:
            """A calculator server."""

            @tool()
            def add(self, a: int, b: int) -> int:
                """Add two numbers."""
                return a + b

            @tool()
            def multiply(self, a: int, b: int) -> int:
                """Multiply two numbers."""
                return a * b

        # Check class has get_server method
        assert hasattr(Calculator, 'get_server')
        assert hasattr(Calculator, '_mcp_server')

        # Get server
        server = Calculator.get_server()
        assert isinstance(server, SimplyMCPServer)
        assert server.config.server.name == "calculator"
        assert server.config.server.version == "1.0.0"

        # Check tools are registered
        add_tool = server.registry.get_tool('add')
        assert add_tool is not None
        assert add_tool['name'] == 'add'

        multiply_tool = server.registry.get_tool('multiply')
        assert multiply_tool is not None
        assert multiply_tool['name'] == 'multiply'

    def test_mcp_server_decorator_mixed_components(self, reset_global_server):
        """Test @mcp_server with tools, prompts, and resources."""
        @mcp_server(name="assistant", version="2.0.0")
        class Assistant:
            """An assistant server."""

            @tool()
            def search(self, query: str) -> list:
                """Search for information."""
                return ["result1", "result2"]

            @prompt()
            def help_prompt(self) -> str:
                """Generate help prompt."""
                return "How can I help you?"

            @resource(uri="config://assistant")
            def get_config(self) -> dict:
                """Get assistant configuration."""
                return {"mode": "helpful"}

        server = Assistant.get_server()

        # Check all components are registered
        assert server.registry.get_tool('search') is not None
        assert server.registry.get_prompt('help_prompt') is not None
        assert server.registry.get_resource('config://assistant') is not None

    def test_mcp_server_decorator_default_name(self, reset_global_server):
        """Test @mcp_server with default name from class name."""
        @mcp_server()
        class MyServer:
            @tool()
            def test(self) -> str:
                return "test"

        server = MyServer.get_server()
        assert server.config.server.name == "MyServer"

    def test_mcp_server_decorator_custom_config(self, reset_global_server):
        """Test @mcp_server with custom configuration."""
        custom_config = SimplyMCPConfig(
            server=ServerMetadataModel(
                name="custom-server",
                version="3.0.0",
                description="Custom server",
                author="Test Author"
            )
        )

        @mcp_server(config=custom_config)
        class CustomServer:
            @tool()
            def test(self) -> str:
                return "test"

        server = CustomServer.get_server()
        assert server.config.server.name == "custom-server"
        assert server.config.server.version == "3.0.0"
        assert server.config.server.author == "Test Author"

    def test_mcp_server_decorator_with_description(self, reset_global_server):
        """Test @mcp_server extracts description from docstring."""
        @mcp_server(name="test-server", version="1.0.0")
        class ServerWithDoc:
            """This is a test server.

            It has multiple lines of documentation.
            """

            @tool()
            def test(self) -> str:
                return "test"

        server = ServerWithDoc.get_server()
        assert server.config.server.description == "This is a test server."

    def test_mcp_server_decorator_method_binding(self, reset_global_server):
        """Test that methods are properly bound in @mcp_server."""
        # Import here to ensure fresh state
        from simply_mcp.api.decorators import mcp_server, tool as tool_dec

        @mcp_server(name="stateful", version="1.0.0")
        class StatefulServer:
            def __init__(self):
                self.counter = 0

            @tool_dec()
            def increment(self) -> int:
                """Increment counter."""
                self.counter += 1
                return self.counter

        server = StatefulServer.get_server()
        tool_config = server.registry.get_tool('increment')
        assert tool_config is not None

        # The handler should be a bound method
        handler = tool_config['handler']
        result1 = handler()
        result2 = handler()
        assert result1 == 1
        assert result2 == 2  # State is preserved


class TestGlobalServerManagement:
    """Tests for global server management."""

    def test_get_global_server_creates_server(self, reset_global_server):
        """Test get_global_server creates a server if none exists."""
        server = get_global_server()
        assert isinstance(server, SimplyMCPServer)
        assert server.config.server.name == "simply-mcp-server"

    def test_get_global_server_returns_same_instance(self, reset_global_server):
        """Test get_global_server returns the same instance."""
        server1 = get_global_server()
        server2 = get_global_server()
        assert server1 is server2

    def test_set_global_server(self, reset_global_server):
        """Test set_global_server sets custom server."""
        custom_config = SimplyMCPConfig(
            server=ServerMetadataModel(
                name="custom-global-server",
                version="1.0.0"
            )
        )
        custom_server = SimplyMCPServer(custom_config)

        set_global_server(custom_server)

        server = get_global_server()
        assert server is custom_server
        assert server.config.server.name == "custom-global-server"


class TestIntegration:
    """Integration tests combining multiple decorators."""

    def test_multiple_decorated_functions(self, reset_global_server):
        """Test multiple functions with different decorators."""
        @tool()
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        @tool()
        def subtract(a: int, b: int) -> int:
            """Subtract numbers."""
            return a - b

        @prompt()
        def greeting(name: str) -> str:
            """Generate greeting."""
            return f"Hello, {name}!"

        @resource(uri="config://app")
        def config() -> dict:
            """Get config."""
            return {"version": "1.0.0"}

        server = get_global_server()

        # Check all registered
        assert server.registry.get_tool('add') is not None
        assert server.registry.get_tool('subtract') is not None
        assert server.registry.get_prompt('greeting') is not None
        assert server.registry.get_resource('config://app') is not None

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_and_regular_tools(self, reset_global_server):
        """Test mixing Pydantic and regular tools."""
        class SearchQuery(BaseModel):
            query: str
            limit: int = 10

        @tool(input_schema=SearchQuery)
        def search(input: SearchQuery) -> list:
            """Search."""
            return []

        @tool()
        def simple(x: int) -> int:
            """Simple tool."""
            return x * 2

        server = get_global_server()
        assert server.registry.get_tool('search') is not None
        assert server.registry.get_tool('simple') is not None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_with_no_parameters(self, reset_global_server):
        """Test @tool on function with no parameters."""
        @tool()
        def no_params() -> str:
            """No parameters."""
            return "result"

        config = no_params._mcp_tool_config
        schema = config['input_schema']
        assert schema['type'] == 'object'
        assert len(schema['properties']) == 0

    def test_tool_with_complex_types(self, reset_global_server):
        """Test @tool with complex type hints."""
        @tool()
        def complex_types(
            items: list[str],
            mapping: dict[str, int],
            optional: Optional[str] = None
        ) -> dict:
            """Complex types."""
            return {}

        config = complex_types._mcp_tool_config
        schema = config['input_schema']
        assert 'items' in schema['properties']
        assert 'mapping' in schema['properties']
        assert 'optional' in schema['properties']

    def test_prompt_with_varargs_ignored(self, reset_global_server):
        """Test @prompt ignores *args and **kwargs."""
        @prompt()
        def variable_args(name: str, *args, **kwargs) -> str:
            """Variable args."""
            return f"Hello, {name}"

        config = variable_args._mcp_prompt_config
        # Should only include 'name', not args/kwargs
        assert config['arguments'] == ['name']

    def test_decorated_async_function(self, reset_global_server):
        """Test decorators work with async functions."""
        @tool()
        async def async_tool(x: int) -> int:
            """Async tool."""
            return x * 2

        config = async_tool._mcp_tool_config
        assert config['name'] == 'async_tool'
        assert config['handler'] == async_tool
