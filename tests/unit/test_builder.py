"""Unit tests for builder API.

Tests the SimplyMCP builder class, including:
- Initialization with various configurations
- add_tool/tool decorator with various configurations
- add_prompt/prompt decorator with various configurations
- add_resource/resource decorator with various configurations
- Method chaining
- Configuration methods
- Pydantic integration
- Auto-schema generation
- Lifecycle methods
- Component queries
- Error cases
"""

import pytest
from typing import Optional

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from simply_mcp.api.builder import SimplyMCP
from simply_mcp.core.config import SimplyMCPConfig, ServerMetadataModel
from simply_mcp.core.errors import ValidationError


class TestSimplyMCPInitialization:
    """Tests for SimplyMCP initialization."""

    def test_default_initialization(self):
        """Test SimplyMCP initialization with defaults."""
        mcp = SimplyMCP()
        assert mcp.name == "simply-mcp-server"
        assert mcp.version == "0.1.0"
        assert mcp.description is None
        assert mcp.server is not None
        assert mcp.config is not None

    def test_initialization_with_name_version(self):
        """Test SimplyMCP initialization with custom name and version."""
        mcp = SimplyMCP(name="my-server", version="1.0.0")
        assert mcp.name == "my-server"
        assert mcp.version == "1.0.0"
        assert mcp.config.server.name == "my-server"
        assert mcp.config.server.version == "1.0.0"

    def test_initialization_with_description(self):
        """Test SimplyMCP initialization with description."""
        mcp = SimplyMCP(
            name="test-server",
            version="2.0.0",
            description="Test server description"
        )
        assert mcp.description == "Test server description"
        assert mcp.config.server.description == "Test server description"

    def test_initialization_with_custom_config(self):
        """Test SimplyMCP initialization with custom config."""
        custom_config = SimplyMCPConfig(
            server=ServerMetadataModel(
                name="custom-server",
                version="3.0.0",
                description="Custom config",
                author="Test Author"
            )
        )

        mcp = SimplyMCP(config=custom_config)
        assert mcp.config.server.name == "custom-server"
        assert mcp.config.server.version == "3.0.0"
        assert mcp.config.server.author == "Test Author"


class TestAddTool:
    """Tests for add_tool method."""

    def test_add_tool_basic(self):
        """Test add_tool with basic function."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = mcp.add_tool("add", add)

        # Check method chaining
        assert result is mcp

        # Check tool is registered
        assert "add" in mcp.list_tools()

        # Check tool config
        tool = mcp.server.registry.get_tool("add")
        assert tool is not None
        assert tool["name"] == "add"
        assert tool["description"] == "Add two numbers."
        assert tool["handler"] == add

    def test_add_tool_custom_description(self):
        """Test add_tool with custom description."""
        mcp = SimplyMCP()

        def multiply(a: int, b: int) -> int:
            return a * b

        mcp.add_tool("multiply", multiply, description="Custom multiply")

        tool = mcp.server.registry.get_tool("multiply")
        assert tool["description"] == "Custom multiply"

    def test_add_tool_auto_description_from_docstring(self):
        """Test add_tool extracts description from docstring."""
        mcp = SimplyMCP()

        def subtract(a: int, b: int) -> int:
            """Subtract b from a."""
            return a - b

        mcp.add_tool("subtract", subtract)

        tool = mcp.server.registry.get_tool("subtract")
        assert tool["description"] == "Subtract b from a."

    def test_add_tool_auto_schema_generation(self):
        """Test add_tool generates schema from function signature."""
        mcp = SimplyMCP()

        def divide(a: float, b: float) -> float:
            """Divide a by b."""
            return a / b

        mcp.add_tool("divide", divide)

        tool = mcp.server.registry.get_tool("divide")
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "number"
        assert schema["properties"]["b"]["type"] == "number"

    def test_add_tool_explicit_schema(self):
        """Test add_tool with explicit schema."""
        mcp = SimplyMCP()

        def process(x: int) -> int:
            return x * 2

        schema = {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 0}
            },
            "required": ["x"]
        }

        mcp.add_tool("process", process, input_schema=schema)

        tool = mcp.server.registry.get_tool("process")
        assert tool["input_schema"] == schema

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_add_tool_pydantic_model(self):
        """Test add_tool with Pydantic model."""
        mcp = SimplyMCP()

        class SearchInput(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, ge=1, le=100)

        def search(input: SearchInput) -> list:
            """Search with validation."""
            return [f"Result for {input.query}"]

        mcp.add_tool("search", search, input_schema=SearchInput)

        tool = mcp.server.registry.get_tool("search")
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]

    def test_add_tool_duplicate_raises_error(self):
        """Test add_tool raises error for duplicate tool name."""
        mcp = SimplyMCP()

        def tool1(x: int) -> int:
            return x

        def tool2(x: int) -> int:
            return x * 2

        mcp.add_tool("duplicate", tool1)

        with pytest.raises(ValidationError, match="already registered"):
            mcp.add_tool("duplicate", tool2)


class TestToolDecorator:
    """Tests for @mcp.tool decorator."""

    def test_tool_decorator_basic(self):
        """Test @mcp.tool() decorator."""
        mcp = SimplyMCP()

        @mcp.tool()
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert "add" in mcp.list_tools()
        assert add.__name__ == "add"
        assert add.__doc__ == "Add two numbers."

    def test_tool_decorator_custom_name(self):
        """Test @mcp.tool with custom name."""
        mcp = SimplyMCP()

        @mcp.tool(name="custom_multiply")
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        assert "custom_multiply" in mcp.list_tools()
        assert "multiply" not in mcp.list_tools()

    def test_tool_decorator_custom_description(self):
        """Test @mcp.tool with custom description."""
        mcp = SimplyMCP()

        @mcp.tool(description="Custom description")
        def divide(a: float, b: float) -> float:
            return a / b

        tool = mcp.server.registry.get_tool("divide")
        assert tool["description"] == "Custom description"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_tool_decorator_pydantic_schema(self):
        """Test @mcp.tool with Pydantic schema."""
        mcp = SimplyMCP()

        class UserInput(BaseModel):
            name: str
            age: int = Field(ge=0, le=150)

        @mcp.tool(input_schema=UserInput)
        def create_user(input: UserInput) -> dict:
            """Create a user."""
            return {"name": input.name, "age": input.age}

        tool = mcp.server.registry.get_tool("create_user")
        assert "name" in tool["input_schema"]["properties"]
        assert "age" in tool["input_schema"]["properties"]


class TestAddPrompt:
    """Tests for add_prompt method."""

    def test_add_prompt_basic(self):
        """Test add_prompt with basic function."""
        mcp = SimplyMCP()

        def greet(name: str) -> str:
            """Generate a greeting."""
            return f"Hello, {name}!"

        result = mcp.add_prompt("greet", greet)

        # Check method chaining
        assert result is mcp

        # Check prompt is registered
        assert "greet" in mcp.list_prompts()

        # Check prompt config
        prompt = mcp.server.registry.get_prompt("greet")
        assert prompt is not None
        assert prompt["name"] == "greet"
        assert prompt["description"] == "Generate a greeting."
        assert prompt["arguments"] == ["name"]
        assert prompt["handler"] == greet

    def test_add_prompt_custom_description(self):
        """Test add_prompt with custom description."""
        mcp = SimplyMCP()

        def code_review(language: str) -> str:
            return f"Review this {language} code..."

        mcp.add_prompt("code_review", code_review, description="Custom code review")

        prompt = mcp.server.registry.get_prompt("code_review")
        assert prompt["description"] == "Custom code review"

    def test_add_prompt_explicit_arguments(self):
        """Test add_prompt with explicit arguments."""
        mcp = SimplyMCP()

        def generate_text(topic: str, style: str = "formal") -> str:
            return f"Write about {topic} in {style} style"

        mcp.add_prompt("generate_text", generate_text, arguments=["topic"])

        prompt = mcp.server.registry.get_prompt("generate_text")
        assert prompt["arguments"] == ["topic"]

    def test_add_prompt_no_arguments(self):
        """Test add_prompt with no arguments."""
        mcp = SimplyMCP()

        def static_prompt() -> str:
            """Static prompt."""
            return "This is a static prompt"

        mcp.add_prompt("static", static_prompt)

        prompt = mcp.server.registry.get_prompt("static")
        # When there are no arguments, the list should be empty or the key may not be present
        assert prompt.get("arguments", []) == []

    def test_add_prompt_duplicate_raises_error(self):
        """Test add_prompt raises error for duplicate prompt name."""
        mcp = SimplyMCP()

        def prompt1() -> str:
            return "Prompt 1"

        def prompt2() -> str:
            return "Prompt 2"

        mcp.add_prompt("duplicate", prompt1)

        with pytest.raises(ValidationError, match="already registered"):
            mcp.add_prompt("duplicate", prompt2)


class TestPromptDecorator:
    """Tests for @mcp.prompt decorator."""

    def test_prompt_decorator_basic(self):
        """Test @mcp.prompt() decorator."""
        mcp = SimplyMCP()

        @mcp.prompt()
        def code_review(language: str = "python") -> str:
            """Generate a code review prompt."""
            return f"Review this {language} code..."

        assert "code_review" in mcp.list_prompts()
        prompt = mcp.server.registry.get_prompt("code_review")
        assert prompt["arguments"] == ["language"]

    def test_prompt_decorator_custom_name(self):
        """Test @mcp.prompt with custom name."""
        mcp = SimplyMCP()

        @mcp.prompt(name="custom_prompt")
        def my_prompt(topic: str) -> str:
            return f"Prompt about {topic}"

        assert "custom_prompt" in mcp.list_prompts()
        assert "my_prompt" not in mcp.list_prompts()

    def test_prompt_decorator_explicit_arguments(self):
        """Test @mcp.prompt with explicit arguments."""
        mcp = SimplyMCP()

        @mcp.prompt(arguments=["topic", "style"])
        def generate_prompt(topic: str, style: str = "formal") -> str:
            return f"Write about {topic} in {style} style"

        prompt = mcp.server.registry.get_prompt("generate_prompt")
        assert prompt["arguments"] == ["topic", "style"]


class TestAddResource:
    """Tests for add_resource method."""

    def test_add_resource_basic(self):
        """Test add_resource with basic function."""
        mcp = SimplyMCP()

        def get_config() -> dict:
            """Get application config."""
            return {"version": "1.0.0"}

        result = mcp.add_resource("config://app", get_config)

        # Check method chaining
        assert result is mcp

        # Check resource is registered
        assert "config://app" in mcp.list_resources()

        # Check resource config
        resource = mcp.server.registry.get_resource("config://app")
        assert resource is not None
        assert resource["uri"] == "config://app"
        assert resource["name"] == "get_config"
        assert resource["description"] == "Get application config."
        assert resource["mime_type"] == "application/json"
        assert resource["handler"] == get_config

    def test_add_resource_custom_name_description(self):
        """Test add_resource with custom name and description."""
        mcp = SimplyMCP()

        def load_data() -> dict:
            return {"data": "value"}

        mcp.add_resource(
            "data://stats",
            load_data,
            name="statistics",
            description="System statistics"
        )

        resource = mcp.server.registry.get_resource("data://stats")
        assert resource["name"] == "statistics"
        assert resource["description"] == "System statistics"

    def test_add_resource_custom_mime_type(self):
        """Test add_resource with custom MIME type."""
        mcp = SimplyMCP()

        def get_text() -> str:
            """Get text data."""
            return "Plain text content"

        mcp.add_resource("data://text", get_text, mime_type="text/plain")

        resource = mcp.server.registry.get_resource("data://text")
        assert resource["mime_type"] == "text/plain"

    def test_add_resource_empty_uri_raises_error(self):
        """Test add_resource raises error for empty URI."""
        mcp = SimplyMCP()

        def bad_resource() -> dict:
            return {}

        with pytest.raises(ValueError, match="Resource URI is required"):
            mcp.add_resource("", bad_resource)

    def test_add_resource_duplicate_raises_error(self):
        """Test add_resource raises error for duplicate URI."""
        mcp = SimplyMCP()

        def resource1() -> dict:
            return {"v": 1}

        def resource2() -> dict:
            return {"v": 2}

        mcp.add_resource("test://resource", resource1)

        with pytest.raises(ValidationError, match="already registered"):
            mcp.add_resource("test://resource", resource2)


class TestResourceDecorator:
    """Tests for @mcp.resource decorator."""

    def test_resource_decorator_basic(self):
        """Test @mcp.resource() decorator."""
        mcp = SimplyMCP()

        @mcp.resource(uri="config://app")
        def get_config() -> dict:
            """Get application config."""
            return {"version": "1.0.0"}

        assert "config://app" in mcp.list_resources()

    def test_resource_decorator_custom_name(self):
        """Test @mcp.resource with custom name."""
        mcp = SimplyMCP()

        @mcp.resource(uri="data://stats", name="statistics")
        def get_stats() -> dict:
            return {"cpu": "50%"}

        resource = mcp.server.registry.get_resource("data://stats")
        assert resource["name"] == "statistics"

    def test_resource_decorator_custom_mime_type(self):
        """Test @mcp.resource with custom MIME type."""
        mcp = SimplyMCP()

        @mcp.resource(uri="data://text", mime_type="text/plain")
        def get_text() -> str:
            return "Plain text"

        resource = mcp.server.registry.get_resource("data://text")
        assert resource["mime_type"] == "text/plain"

    def test_resource_decorator_empty_uri_raises_error(self):
        """Test @mcp.resource raises error for empty URI."""
        mcp = SimplyMCP()

        with pytest.raises(ValueError, match="Resource URI is required"):
            @mcp.resource(uri="")
            def bad_resource() -> dict:
                return {}


class TestMethodChaining:
    """Tests for method chaining."""

    def test_method_chaining_add_tool(self):
        """Test method chaining with add_tool."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        result = mcp.add_tool("add", add).add_tool("multiply", multiply)

        assert result is mcp
        assert len(mcp.list_tools()) == 2

    def test_method_chaining_mixed_components(self):
        """Test method chaining with mixed components."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            return a + b

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def get_config() -> dict:
            return {"version": "1.0.0"}

        result = (
            mcp.add_tool("add", add)
            .add_prompt("greet", greet)
            .add_resource("config://app", get_config)
        )

        assert result is mcp
        assert len(mcp.list_tools()) == 1
        assert len(mcp.list_prompts()) == 1
        assert len(mcp.list_resources()) == 1

    def test_method_chaining_with_configure(self):
        """Test method chaining with configure."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            return a + b

        result = mcp.add_tool("add", add).configure(port=3000, log_level="DEBUG")

        assert result is mcp
        assert mcp.config.transport.port == 3000
        assert mcp.config.logging.level == "DEBUG"


class TestConfigure:
    """Tests for configure method."""

    def test_configure_port(self):
        """Test configure with port."""
        mcp = SimplyMCP()
        mcp.configure(port=8080)

        assert mcp.config.transport.port == 8080

    def test_configure_log_level(self):
        """Test configure with log level."""
        mcp = SimplyMCP()
        mcp.configure(log_level="WARNING")

        assert mcp.config.logging.level == "WARNING"

    def test_configure_both_port_and_log_level(self):
        """Test configure with both port and log level."""
        mcp = SimplyMCP()
        mcp.configure(port=9000, log_level="ERROR")

        assert mcp.config.transport.port == 9000
        assert mcp.config.logging.level == "ERROR"

    def test_configure_invalid_log_level(self):
        """Test configure raises error for invalid log level."""
        mcp = SimplyMCP()

        with pytest.raises(ValueError, match="Invalid log level"):
            mcp.configure(log_level="INVALID")

    def test_configure_returns_self(self):
        """Test configure returns self for chaining."""
        mcp = SimplyMCP()
        result = mcp.configure(port=3000)

        assert result is mcp


class TestLifecycleMethods:
    """Tests for lifecycle methods."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialize method."""
        mcp = SimplyMCP()
        result = await mcp.initialize()

        assert result is mcp
        assert mcp.server.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_returns_self(self):
        """Test initialize returns self for chaining."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            return a + b

        result = await mcp.add_tool("add", add).initialize()

        assert result is mcp

    def test_run_stdio_not_initialized_raises_error(self):
        """Test run_stdio raises error if not initialized."""
        mcp = SimplyMCP()

        with pytest.raises(RuntimeError, match="not initialized"):
            import asyncio
            asyncio.run(mcp.run_stdio())

    def test_run_unsupported_transport_raises_error(self):
        """Test run raises error for unsupported transport."""
        mcp = SimplyMCP()

        with pytest.raises(ValueError, match="Unsupported transport"):
            import asyncio
            asyncio.run(mcp.run("http"))


class TestServerAccess:
    """Tests for get_server method."""

    def test_get_server(self):
        """Test get_server returns underlying server."""
        mcp = SimplyMCP()
        server = mcp.get_server()

        from simply_mcp.core.server import SimplyMCPServer
        assert isinstance(server, SimplyMCPServer)
        assert server is mcp.server


class TestComponentQueries:
    """Tests for component query methods."""

    def test_list_tools_empty(self):
        """Test list_tools returns empty list initially."""
        mcp = SimplyMCP()
        assert mcp.list_tools() == []

    def test_list_tools_with_tools(self):
        """Test list_tools returns registered tools."""
        mcp = SimplyMCP()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        mcp.add_tool("add", add)
        mcp.add_tool("multiply", multiply)

        tools = mcp.list_tools()
        assert len(tools) == 2
        assert "add" in tools
        assert "multiply" in tools

    def test_list_prompts_empty(self):
        """Test list_prompts returns empty list initially."""
        mcp = SimplyMCP()
        assert mcp.list_prompts() == []

    def test_list_prompts_with_prompts(self):
        """Test list_prompts returns registered prompts."""
        mcp = SimplyMCP()

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def farewell(name: str) -> str:
            return f"Goodbye, {name}!"

        mcp.add_prompt("greet", greet)
        mcp.add_prompt("farewell", farewell)

        prompts = mcp.list_prompts()
        assert len(prompts) == 2
        assert "greet" in prompts
        assert "farewell" in prompts

    def test_list_resources_empty(self):
        """Test list_resources returns empty list initially."""
        mcp = SimplyMCP()
        assert mcp.list_resources() == []

    def test_list_resources_with_resources(self):
        """Test list_resources returns registered resources."""
        mcp = SimplyMCP()

        def get_config() -> dict:
            return {"version": "1.0.0"}

        def get_stats() -> dict:
            return {"cpu": "50%"}

        mcp.add_resource("config://app", get_config)
        mcp.add_resource("data://stats", get_stats)

        resources = mcp.list_resources()
        assert len(resources) == 2
        assert "config://app" in resources
        assert "data://stats" in resources


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tool_with_no_parameters(self):
        """Test add_tool with function that has no parameters."""
        mcp = SimplyMCP()

        def no_params() -> str:
            """No parameters."""
            return "result"

        mcp.add_tool("no_params", no_params)

        tool = mcp.server.registry.get_tool("no_params")
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert len(schema["properties"]) == 0

    def test_tool_with_optional_parameters(self):
        """Test add_tool with optional parameters."""
        mcp = SimplyMCP()

        def with_optional(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        mcp.add_tool("greet", with_optional)

        tool = mcp.server.registry.get_tool("greet")
        schema = tool["input_schema"]
        assert "name" in schema["required"]
        assert "greeting" not in schema.get("required", [])

    def test_prompt_with_no_arguments(self):
        """Test add_prompt with function that has no arguments."""
        mcp = SimplyMCP()

        def static_prompt() -> str:
            return "Static prompt text"

        mcp.add_prompt("static", static_prompt)

        prompt = mcp.server.registry.get_prompt("static")
        assert prompt.get("arguments", []) == []

    def test_tool_without_docstring(self):
        """Test add_tool with function without docstring."""
        mcp = SimplyMCP()

        def no_doc(x: int) -> int:
            return x * 2

        mcp.add_tool("no_doc", no_doc)

        tool = mcp.server.registry.get_tool("no_doc")
        assert tool["description"] == "Tool: no_doc"

    def test_async_tool_function(self):
        """Test add_tool with async function."""
        mcp = SimplyMCP()

        async def async_tool(x: int) -> int:
            """Async tool."""
            return x * 2

        mcp.add_tool("async_tool", async_tool)

        tool = mcp.server.registry.get_tool("async_tool")
        assert tool["name"] == "async_tool"
        assert tool["handler"] == async_tool


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_complete_workflow(self):
        """Test complete workflow with all features."""
        mcp = SimplyMCP(name="test-server", version="1.0.0")

        # Add tools
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @mcp.tool()
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        mcp.add_tool("add", add)

        # Add prompts
        @mcp.prompt()
        def code_review(language: str = "python") -> str:
            """Generate code review prompt."""
            return f"Review this {language} code..."

        # Add resources
        @mcp.resource(uri="config://app")
        def get_config() -> dict:
            """Get application config."""
            return {"version": "1.0.0"}

        # Configure
        mcp.configure(port=3000, log_level="DEBUG")

        # Verify everything
        assert len(mcp.list_tools()) == 2
        assert len(mcp.list_prompts()) == 1
        assert len(mcp.list_resources()) == 1
        assert mcp.config.transport.port == 3000
        assert mcp.config.logging.level == "DEBUG"

    @pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
    def test_pydantic_and_regular_tools(self):
        """Test mixing Pydantic and regular tools."""
        mcp = SimplyMCP()

        class SearchQuery(BaseModel):
            query: str = Field(description="Search query")
            limit: int = Field(default=10, ge=1, le=100)

        @mcp.tool(input_schema=SearchQuery)
        def search(input: SearchQuery) -> list:
            """Search with Pydantic validation."""
            return [f"Result for {input.query}"]

        @mcp.tool()
        def simple(x: int) -> int:
            """Simple tool with auto-schema."""
            return x * 2

        assert len(mcp.list_tools()) == 2
        search_tool = mcp.server.registry.get_tool("search")
        assert "query" in search_tool["input_schema"]["properties"]

    def test_method_chaining_fluent_api(self):
        """Test fluent API with extensive method chaining."""
        mcp = SimplyMCP(name="calc", version="1.0.0")

        def add(a: int, b: int) -> int:
            return a + b

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        def get_config() -> dict:
            return {"version": "1.0.0"}

        # Chain everything
        result = (
            mcp.add_tool("add", add, description="Add numbers")
            .add_prompt("greet", greet, arguments=["name"])
            .add_resource("config://app", get_config, mime_type="application/json")
            .configure(port=3000, log_level="INFO")
        )

        assert result is mcp
        assert len(mcp.list_tools()) == 1
        assert len(mcp.list_prompts()) == 1
        assert len(mcp.list_resources()) == 1
        assert mcp.config.transport.port == 3000
