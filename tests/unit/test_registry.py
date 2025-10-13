"""Unit tests for component registry system."""

import threading
from typing import Any

import pytest

from simply_mcp.core.errors import ValidationError
from simply_mcp.core.registry import ComponentRegistry
from simply_mcp.core.types import PromptConfigModel, ResourceConfigModel, ToolConfigModel


# Test fixtures


@pytest.fixture
def registry() -> ComponentRegistry:
    """Create a fresh registry for each test."""
    return ComponentRegistry()


@pytest.fixture
def sample_tool_config() -> ToolConfigModel:
    """Create a sample tool configuration."""
    return ToolConfigModel(
        name="add",
        description="Add two numbers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        handler=lambda a, b: a + b,
    )


@pytest.fixture
def sample_prompt_config() -> PromptConfigModel:
    """Create a sample prompt configuration."""
    return PromptConfigModel(
        name="greeting",
        description="Generate a greeting message",
        template="Hello, {name}!",
        arguments=["name"],
    )


@pytest.fixture
def sample_resource_config() -> ResourceConfigModel:
    """Create a sample resource configuration."""
    return ResourceConfigModel(
        uri="file:///data/config.json",
        name="config",
        description="Configuration file",
        mime_type="application/json",
        handler=lambda: {"key": "value"},
    )


# Test classes


class TestRegistryInitialization:
    """Tests for registry initialization."""

    def test_registry_creation(self, registry: ComponentRegistry) -> None:
        """Test creating a registry instance."""
        assert isinstance(registry, ComponentRegistry)

    def test_registry_starts_empty(self, registry: ComponentRegistry) -> None:
        """Test that registry starts with no registered components."""
        assert len(registry.list_tools()) == 0
        assert len(registry.list_prompts()) == 0
        assert len(registry.list_resources()) == 0

    def test_registry_stats_starts_at_zero(self, registry: ComponentRegistry) -> None:
        """Test that registry statistics start at zero."""
        stats = registry.get_stats()
        assert stats["tools"] == 0
        assert stats["prompts"] == 0
        assert stats["resources"] == 0
        assert stats["total"] == 0


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_tool(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test registering a tool."""
        registry.register_tool(sample_tool_config)
        assert registry.has_tool("add")

    def test_register_multiple_tools(self, registry: ComponentRegistry) -> None:
        """Test registering multiple tools."""
        tool1 = ToolConfigModel(
            name="add",
            description="Add numbers",
            input_schema={"type": "object"},
            handler=lambda a, b: a + b,
        )
        tool2 = ToolConfigModel(
            name="multiply",
            description="Multiply numbers",
            input_schema={"type": "object"},
            handler=lambda a, b: a * b,
        )

        registry.register_tool(tool1)
        registry.register_tool(tool2)

        assert registry.has_tool("add")
        assert registry.has_tool("multiply")
        assert len(registry.list_tools()) == 2

    def test_register_duplicate_tool(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test that registering a duplicate tool raises ValidationError."""
        registry.register_tool(sample_tool_config)

        with pytest.raises(ValidationError) as exc_info:
            registry.register_tool(sample_tool_config)

        assert exc_info.value.code == "DUPLICATE_TOOL"
        assert "add" in exc_info.value.message
        assert exc_info.value.context["tool_name"] == "add"

    def test_register_tool_case_insensitive(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test that tool registration is case-insensitive."""
        registry.register_tool(sample_tool_config)

        # Try to register with different case
        duplicate_config = ToolConfigModel(
            name="ADD",
            description="Add numbers",
            input_schema={"type": "object"},
            handler=lambda a, b: a + b,
        )

        with pytest.raises(ValidationError) as exc_info:
            registry.register_tool(duplicate_config)

        assert exc_info.value.code == "DUPLICATE_TOOL"

    def test_register_tool_with_metadata(self, registry: ComponentRegistry) -> None:
        """Test registering a tool with metadata."""
        tool_config = ToolConfigModel(
            name="calculate",
            description="Perform calculations",
            input_schema={"type": "object"},
            handler=lambda: None,
            metadata={"version": "1.0", "author": "test"},
        )

        registry.register_tool(tool_config)
        retrieved = registry.get_tool("calculate")

        assert retrieved is not None
        assert hasattr(retrieved, "metadata")
        assert retrieved.metadata["version"] == "1.0"
        assert retrieved.metadata["author"] == "test"


class TestPromptRegistration:
    """Tests for prompt registration."""

    def test_register_prompt(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test registering a prompt."""
        registry.register_prompt(sample_prompt_config)
        assert registry.has_prompt("greeting")

    def test_register_multiple_prompts(self, registry: ComponentRegistry) -> None:
        """Test registering multiple prompts."""
        prompt1 = PromptConfigModel(
            name="greeting",
            description="Greet user",
            template="Hello, {name}!",
        )
        prompt2 = PromptConfigModel(
            name="farewell",
            description="Say goodbye",
            template="Goodbye, {name}!",
        )

        registry.register_prompt(prompt1)
        registry.register_prompt(prompt2)

        assert registry.has_prompt("greeting")
        assert registry.has_prompt("farewell")
        assert len(registry.list_prompts()) == 2

    def test_register_duplicate_prompt(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test that registering a duplicate prompt raises ValidationError."""
        registry.register_prompt(sample_prompt_config)

        with pytest.raises(ValidationError) as exc_info:
            registry.register_prompt(sample_prompt_config)

        assert exc_info.value.code == "DUPLICATE_PROMPT"
        assert "greeting" in exc_info.value.message
        assert exc_info.value.context["prompt_name"] == "greeting"

    def test_register_prompt_case_insensitive(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test that prompt registration is case-insensitive."""
        registry.register_prompt(sample_prompt_config)

        # Try to register with different case
        duplicate_config = PromptConfigModel(
            name="GREETING",
            description="Greet user",
            template="Hello!",
        )

        with pytest.raises(ValidationError) as exc_info:
            registry.register_prompt(duplicate_config)

        assert exc_info.value.code == "DUPLICATE_PROMPT"

    def test_register_prompt_with_handler(self, registry: ComponentRegistry) -> None:
        """Test registering a prompt with a handler function."""
        prompt_config = PromptConfigModel(
            name="dynamic",
            description="Dynamic prompt",
            handler=lambda name: f"Hello, {name}!",
        )

        registry.register_prompt(prompt_config)
        retrieved = registry.get_prompt("dynamic")

        assert retrieved is not None
        assert hasattr(retrieved, "handler")
        assert callable(retrieved.handler)


class TestResourceRegistration:
    """Tests for resource registration."""

    def test_register_resource(
        self, registry: ComponentRegistry, sample_resource_config: ResourceConfigModel
    ) -> None:
        """Test registering a resource."""
        registry.register_resource(sample_resource_config)
        assert registry.has_resource("file:///data/config.json")

    def test_register_multiple_resources(self, registry: ComponentRegistry) -> None:
        """Test registering multiple resources."""
        resource1 = ResourceConfigModel(
            uri="file:///data/config.json",
            name="config",
            description="Config file",
            mime_type="application/json",
            handler=lambda: {},
        )
        resource2 = ResourceConfigModel(
            uri="file:///data/schema.json",
            name="schema",
            description="Schema file",
            mime_type="application/json",
            handler=lambda: {},
        )

        registry.register_resource(resource1)
        registry.register_resource(resource2)

        assert registry.has_resource("file:///data/config.json")
        assert registry.has_resource("file:///data/schema.json")
        assert len(registry.list_resources()) == 2

    def test_register_duplicate_resource(
        self, registry: ComponentRegistry, sample_resource_config: ResourceConfigModel
    ) -> None:
        """Test that registering a duplicate resource raises ValidationError."""
        registry.register_resource(sample_resource_config)

        with pytest.raises(ValidationError) as exc_info:
            registry.register_resource(sample_resource_config)

        assert exc_info.value.code == "DUPLICATE_RESOURCE"
        assert "file:///data/config.json" in exc_info.value.message
        assert exc_info.value.context["resource_uri"] == "file:///data/config.json"

    def test_register_resource_uri_exact_match(self, registry: ComponentRegistry) -> None:
        """Test that resource URIs must match exactly (case-sensitive)."""
        resource1 = ResourceConfigModel(
            uri="file:///data/config.json",
            name="config",
            description="Config file",
            mime_type="application/json",
            handler=lambda: {},
        )
        resource2 = ResourceConfigModel(
            uri="file:///data/Config.json",
            name="config2",
            description="Config file 2",
            mime_type="application/json",
            handler=lambda: {},
        )

        registry.register_resource(resource1)
        registry.register_resource(resource2)  # Should succeed, different URI

        assert registry.has_resource("file:///data/config.json")
        assert registry.has_resource("file:///data/Config.json")
        assert len(registry.list_resources()) == 2


class TestQueryMethods:
    """Tests for query methods."""

    def test_get_tool(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test getting a tool by name."""
        registry.register_tool(sample_tool_config)
        tool = registry.get_tool("add")

        assert tool is not None
        assert tool.name == "add"
        assert tool.description == "Add two numbers"

    def test_get_tool_case_insensitive(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test that tool lookup is case-insensitive."""
        registry.register_tool(sample_tool_config)

        assert registry.get_tool("add") is not None
        assert registry.get_tool("ADD") is not None
        assert registry.get_tool("Add") is not None

    def test_get_tool_not_found(self, registry: ComponentRegistry) -> None:
        """Test getting a non-existent tool."""
        tool = registry.get_tool("nonexistent")
        assert tool is None

    def test_get_prompt(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test getting a prompt by name."""
        registry.register_prompt(sample_prompt_config)
        prompt = registry.get_prompt("greeting")

        assert prompt is not None
        assert prompt.name == "greeting"
        assert prompt.description == "Generate a greeting message"

    def test_get_prompt_case_insensitive(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test that prompt lookup is case-insensitive."""
        registry.register_prompt(sample_prompt_config)

        assert registry.get_prompt("greeting") is not None
        assert registry.get_prompt("GREETING") is not None
        assert registry.get_prompt("Greeting") is not None

    def test_get_prompt_not_found(self, registry: ComponentRegistry) -> None:
        """Test getting a non-existent prompt."""
        prompt = registry.get_prompt("nonexistent")
        assert prompt is None

    def test_get_resource(
        self, registry: ComponentRegistry, sample_resource_config: ResourceConfigModel
    ) -> None:
        """Test getting a resource by URI."""
        registry.register_resource(sample_resource_config)
        resource = registry.get_resource("file:///data/config.json")

        assert resource is not None
        assert resource.uri == "file:///data/config.json"
        assert resource.name == "config"

    def test_get_resource_not_found(self, registry: ComponentRegistry) -> None:
        """Test getting a non-existent resource."""
        resource = registry.get_resource("file:///nonexistent")
        assert resource is None


class TestListMethods:
    """Tests for list methods."""

    def test_list_tools_empty(self, registry: ComponentRegistry) -> None:
        """Test listing tools when registry is empty."""
        tools = registry.list_tools()
        assert isinstance(tools, list)
        assert len(tools) == 0

    def test_list_tools(self, registry: ComponentRegistry) -> None:
        """Test listing multiple tools."""
        tool1 = ToolConfigModel(
            name="add",
            description="Add",
            input_schema={"type": "object"},
            handler=lambda: None,
        )
        tool2 = ToolConfigModel(
            name="subtract",
            description="Subtract",
            input_schema={"type": "object"},
            handler=lambda: None,
        )

        registry.register_tool(tool1)
        registry.register_tool(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2

        tool_names = {tool.name for tool in tools}
        assert "add" in tool_names
        assert "subtract" in tool_names

    def test_list_prompts_empty(self, registry: ComponentRegistry) -> None:
        """Test listing prompts when registry is empty."""
        prompts = registry.list_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) == 0

    def test_list_prompts(self, registry: ComponentRegistry) -> None:
        """Test listing multiple prompts."""
        prompt1 = PromptConfigModel(
            name="greeting",
            description="Greet",
            template="Hello",
        )
        prompt2 = PromptConfigModel(
            name="farewell",
            description="Goodbye",
            template="Bye",
        )

        registry.register_prompt(prompt1)
        registry.register_prompt(prompt2)

        prompts = registry.list_prompts()
        assert len(prompts) == 2

        prompt_names = {prompt.name for prompt in prompts}
        assert "greeting" in prompt_names
        assert "farewell" in prompt_names

    def test_list_resources_empty(self, registry: ComponentRegistry) -> None:
        """Test listing resources when registry is empty."""
        resources = registry.list_resources()
        assert isinstance(resources, list)
        assert len(resources) == 0

    def test_list_resources(self, registry: ComponentRegistry) -> None:
        """Test listing multiple resources."""
        resource1 = ResourceConfigModel(
            uri="file:///data/config.json",
            name="config",
            description="Config",
            mime_type="application/json",
            handler=lambda: {},
        )
        resource2 = ResourceConfigModel(
            uri="file:///data/schema.json",
            name="schema",
            description="Schema",
            mime_type="application/json",
            handler=lambda: {},
        )

        registry.register_resource(resource1)
        registry.register_resource(resource2)

        resources = registry.list_resources()
        assert len(resources) == 2

        resource_uris = {resource.uri for resource in resources}
        assert "file:///data/config.json" in resource_uris
        assert "file:///data/schema.json" in resource_uris


class TestHasMethods:
    """Tests for has_* methods."""

    def test_has_tool_exists(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test has_tool when tool exists."""
        registry.register_tool(sample_tool_config)
        assert registry.has_tool("add") is True

    def test_has_tool_not_exists(self, registry: ComponentRegistry) -> None:
        """Test has_tool when tool doesn't exist."""
        assert registry.has_tool("nonexistent") is False

    def test_has_tool_case_insensitive(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test has_tool is case-insensitive."""
        registry.register_tool(sample_tool_config)
        assert registry.has_tool("add") is True
        assert registry.has_tool("ADD") is True
        assert registry.has_tool("Add") is True

    def test_has_prompt_exists(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test has_prompt when prompt exists."""
        registry.register_prompt(sample_prompt_config)
        assert registry.has_prompt("greeting") is True

    def test_has_prompt_not_exists(self, registry: ComponentRegistry) -> None:
        """Test has_prompt when prompt doesn't exist."""
        assert registry.has_prompt("nonexistent") is False

    def test_has_prompt_case_insensitive(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test has_prompt is case-insensitive."""
        registry.register_prompt(sample_prompt_config)
        assert registry.has_prompt("greeting") is True
        assert registry.has_prompt("GREETING") is True
        assert registry.has_prompt("Greeting") is True

    def test_has_resource_exists(
        self, registry: ComponentRegistry, sample_resource_config: ResourceConfigModel
    ) -> None:
        """Test has_resource when resource exists."""
        registry.register_resource(sample_resource_config)
        assert registry.has_resource("file:///data/config.json") is True

    def test_has_resource_not_exists(self, registry: ComponentRegistry) -> None:
        """Test has_resource when resource doesn't exist."""
        assert registry.has_resource("file:///nonexistent") is False


class TestUnregisterMethods:
    """Tests for unregister methods."""

    def test_unregister_tool_success(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test successfully unregistering a tool."""
        registry.register_tool(sample_tool_config)
        assert registry.has_tool("add")

        result = registry.unregister_tool("add")
        assert result is True
        assert not registry.has_tool("add")

    def test_unregister_tool_not_found(self, registry: ComponentRegistry) -> None:
        """Test unregistering a non-existent tool."""
        result = registry.unregister_tool("nonexistent")
        assert result is False

    def test_unregister_tool_case_insensitive(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test unregistering a tool is case-insensitive."""
        registry.register_tool(sample_tool_config)

        result = registry.unregister_tool("ADD")
        assert result is True
        assert not registry.has_tool("add")

    def test_unregister_prompt_success(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test successfully unregistering a prompt."""
        registry.register_prompt(sample_prompt_config)
        assert registry.has_prompt("greeting")

        result = registry.unregister_prompt("greeting")
        assert result is True
        assert not registry.has_prompt("greeting")

    def test_unregister_prompt_not_found(self, registry: ComponentRegistry) -> None:
        """Test unregistering a non-existent prompt."""
        result = registry.unregister_prompt("nonexistent")
        assert result is False

    def test_unregister_prompt_case_insensitive(
        self, registry: ComponentRegistry, sample_prompt_config: PromptConfigModel
    ) -> None:
        """Test unregistering a prompt is case-insensitive."""
        registry.register_prompt(sample_prompt_config)

        result = registry.unregister_prompt("GREETING")
        assert result is True
        assert not registry.has_prompt("greeting")

    def test_unregister_resource_success(
        self, registry: ComponentRegistry, sample_resource_config: ResourceConfigModel
    ) -> None:
        """Test successfully unregistering a resource."""
        registry.register_resource(sample_resource_config)
        assert registry.has_resource("file:///data/config.json")

        result = registry.unregister_resource("file:///data/config.json")
        assert result is True
        assert not registry.has_resource("file:///data/config.json")

    def test_unregister_resource_not_found(self, registry: ComponentRegistry) -> None:
        """Test unregistering a non-existent resource."""
        result = registry.unregister_resource("file:///nonexistent")
        assert result is False


class TestClearMethod:
    """Tests for clear method."""

    def test_clear_empty_registry(self, registry: ComponentRegistry) -> None:
        """Test clearing an empty registry."""
        registry.clear()
        stats = registry.get_stats()
        assert stats["total"] == 0

    def test_clear_populated_registry(
        self,
        registry: ComponentRegistry,
        sample_tool_config: ToolConfigModel,
        sample_prompt_config: PromptConfigModel,
        sample_resource_config: ResourceConfigModel,
    ) -> None:
        """Test clearing a populated registry."""
        registry.register_tool(sample_tool_config)
        registry.register_prompt(sample_prompt_config)
        registry.register_resource(sample_resource_config)

        assert registry.get_stats()["total"] == 3

        registry.clear()

        assert len(registry.list_tools()) == 0
        assert len(registry.list_prompts()) == 0
        assert len(registry.list_resources()) == 0
        assert registry.get_stats()["total"] == 0

    def test_clear_and_re_register(
        self, registry: ComponentRegistry, sample_tool_config: ToolConfigModel
    ) -> None:
        """Test registering components after clearing."""
        registry.register_tool(sample_tool_config)
        registry.clear()

        # Should be able to re-register after clearing
        registry.register_tool(sample_tool_config)
        assert registry.has_tool("add")


class TestGetStatsMethod:
    """Tests for get_stats method."""

    def test_get_stats_empty(self, registry: ComponentRegistry) -> None:
        """Test get_stats on empty registry."""
        stats = registry.get_stats()
        assert stats["tools"] == 0
        assert stats["prompts"] == 0
        assert stats["resources"] == 0
        assert stats["total"] == 0

    def test_get_stats_with_components(
        self,
        registry: ComponentRegistry,
        sample_tool_config: ToolConfigModel,
        sample_prompt_config: PromptConfigModel,
        sample_resource_config: ResourceConfigModel,
    ) -> None:
        """Test get_stats with registered components."""
        registry.register_tool(sample_tool_config)
        registry.register_prompt(sample_prompt_config)
        registry.register_resource(sample_resource_config)

        stats = registry.get_stats()
        assert stats["tools"] == 1
        assert stats["prompts"] == 1
        assert stats["resources"] == 1
        assert stats["total"] == 3

    def test_get_stats_multiple_components(self, registry: ComponentRegistry) -> None:
        """Test get_stats with multiple components of each type."""
        # Register 3 tools
        for i in range(3):
            tool = ToolConfigModel(
                name=f"tool{i}",
                description=f"Tool {i}",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
            registry.register_tool(tool)

        # Register 2 prompts
        for i in range(2):
            prompt = PromptConfigModel(
                name=f"prompt{i}",
                description=f"Prompt {i}",
                template="Template",
            )
            registry.register_prompt(prompt)

        # Register 1 resource
        resource = ResourceConfigModel(
            uri="file:///data/test.json",
            name="test",
            description="Test",
            mime_type="application/json",
            handler=lambda: {},
        )
        registry.register_resource(resource)

        stats = registry.get_stats()
        assert stats["tools"] == 3
        assert stats["prompts"] == 2
        assert stats["resources"] == 1
        assert stats["total"] == 6


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_tool_registration(self, registry: ComponentRegistry) -> None:
        """Test concurrent tool registration from multiple threads."""
        num_threads = 10
        tools_per_thread = 10
        threads = []

        def register_tools(thread_id: int) -> None:
            for i in range(tools_per_thread):
                tool = ToolConfigModel(
                    name=f"tool_{thread_id}_{i}",
                    description=f"Tool {thread_id}-{i}",
                    input_schema={"type": "object"},
                    handler=lambda: None,
                )
                registry.register_tool(tool)

        for i in range(num_threads):
            thread = threading.Thread(target=register_tools, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        stats = registry.get_stats()
        assert stats["tools"] == num_threads * tools_per_thread

    def test_concurrent_mixed_operations(self, registry: ComponentRegistry) -> None:
        """Test concurrent mixed operations (register, get, list)."""
        num_threads = 5
        threads = []

        def mixed_operations(thread_id: int) -> None:
            # Register
            tool = ToolConfigModel(
                name=f"tool_{thread_id}",
                description=f"Tool {thread_id}",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
            registry.register_tool(tool)

            # Get
            registry.get_tool(f"tool_{thread_id}")

            # List
            registry.list_tools()

            # Has
            registry.has_tool(f"tool_{thread_id}")

        for i in range(num_threads):
            thread = threading.Thread(target=mixed_operations, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert registry.get_stats()["tools"] == num_threads

    def test_concurrent_unregister(self, registry: ComponentRegistry) -> None:
        """Test concurrent unregister operations."""
        # Register tools first
        num_tools = 20
        for i in range(num_tools):
            tool = ToolConfigModel(
                name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
            registry.register_tool(tool)

        assert registry.get_stats()["tools"] == num_tools

        # Unregister from multiple threads
        threads = []

        def unregister_tool(tool_id: int) -> None:
            registry.unregister_tool(f"tool_{tool_id}")

        for i in range(num_tools):
            thread = threading.Thread(target=unregister_tool, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert registry.get_stats()["tools"] == 0

    def test_concurrent_clear(self, registry: ComponentRegistry) -> None:
        """Test concurrent clear operations."""
        # Register some components
        for i in range(10):
            tool = ToolConfigModel(
                name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
            registry.register_tool(tool)

        threads = []

        def clear_registry() -> None:
            registry.clear()

        # Multiple threads trying to clear
        for _ in range(5):
            thread = threading.Thread(target=clear_registry)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert registry.get_stats()["tools"] == 0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_name_handling(self, registry: ComponentRegistry) -> None:
        """Test handling of empty names - should fail validation."""
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            tool = ToolConfigModel(
                name="",
                description="Empty name tool",
                input_schema={"type": "object"},
                handler=lambda: None,
            )

    def test_special_characters_in_names(self, registry: ComponentRegistry) -> None:
        """Test handling of special characters in names."""
        tool = ToolConfigModel(
            name="tool-with-dashes_and_underscores.and.dots",
            description="Special chars",
            input_schema={"type": "object"},
            handler=lambda: None,
        )

        registry.register_tool(tool)
        assert registry.has_tool("tool-with-dashes_and_underscores.and.dots")

    def test_unicode_in_names(self, registry: ComponentRegistry) -> None:
        """Test handling of Unicode characters in names."""
        tool = ToolConfigModel(
            name="计算器",
            description="Calculator in Chinese",
            input_schema={"type": "object"},
            handler=lambda: None,
        )

        registry.register_tool(tool)
        assert registry.has_tool("计算器")
        assert registry.get_tool("计算器") is not None

    def test_resource_uri_edge_cases(self, registry: ComponentRegistry) -> None:
        """Test various URI formats for resources."""
        uris = [
            "file:///path/to/file.json",
            "http://example.com/resource",
            "https://example.com/resource",
            "ftp://server/file",
            "custom://protocol/resource",
        ]

        for uri in uris:
            resource = ResourceConfigModel(
                uri=uri,
                name=f"resource_{uri}",
                description="Test resource",
                mime_type="application/json",
                handler=lambda: {},
            )
            registry.register_resource(resource)

        for uri in uris:
            assert registry.has_resource(uri)

    def test_large_number_of_components(self, registry: ComponentRegistry) -> None:
        """Test registry performance with many components."""
        num_components = 1000

        # Register many tools
        for i in range(num_components):
            tool = ToolConfigModel(
                name=f"tool_{i:04d}",
                description=f"Tool {i}",
                input_schema={"type": "object"},
                handler=lambda: None,
            )
            registry.register_tool(tool)

        stats = registry.get_stats()
        assert stats["tools"] == num_components

        # Verify all are retrievable
        assert registry.has_tool("tool_0000")
        assert registry.has_tool("tool_0500")
        assert registry.has_tool("tool_0999")
