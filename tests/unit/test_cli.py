"""Comprehensive tests for Simply-MCP CLI.

Tests all CLI commands including:
- Main entry point
- Run command with API auto-detection
- Config commands (init, validate, show)
- List command with filters
- Error handling and edge cases
"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from simply_mcp.api.programmatic import BuildMCPServer
from simply_mcp.api.decorators import get_global_server, set_global_server, tool
from simply_mcp.cli.config import config, init, show, validate
from simply_mcp.cli.list_cmd import list_components
from simply_mcp.cli.main import cli
from simply_mcp.cli.run import run
from simply_mcp.cli.utils import (
    create_components_table,
    detect_api_style,
    format_error,
    format_info,
    format_success,
    get_server_instance,
    load_python_module,
    validate_python_file,
)
from simply_mcp.core.config import SimplyMCPConfig, get_default_config
from simply_mcp.core.server import SimplyMCPServer


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_server_file(tmp_path):
    """Create a temporary server file for testing."""
    server_file = tmp_path / "server.py"
    server_file.write_text("""
from simply_mcp import tool, prompt, resource

@tool()
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b

@prompt()
def greet(name: str) -> str:
    '''Generate a greeting.'''
    return f"Hello, {name}!"

@resource(uri="config://test")
def get_config() -> dict:
    '''Get test config.'''
    return {"test": True}
""")
    return server_file


@pytest.fixture
def temp_builder_file(tmp_path):
    """Create a temporary builder API server file."""
    server_file = tmp_path / "builder_server.py"
    server_file.write_text("""
from simply_mcp import BuildMCPServer

mcp = BuildMCPServer(name="test-server", version="1.0.0")

@mcp.tool()
def multiply(a: int, b: int) -> int:
    '''Multiply two numbers.'''
    return a * b
""")
    return server_file


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file."""
    config_file = tmp_path / "test.config.toml"
    config_file.write_text("""
[server]
name = "test-server"
version = "1.0.0"
description = "Test server"

[transport]
type = "stdio"
port = 3000

[logging]
level = "INFO"
format = "json"
""")
    return config_file


# Test CLI Utils


def test_validate_python_file(tmp_path):
    """Test Python file validation."""
    # Valid Python file
    py_file = tmp_path / "test.py"
    py_file.write_text("# test")
    assert validate_python_file(str(py_file))

    # Non-Python file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("test")
    assert not validate_python_file(str(txt_file))

    # Non-existent file
    assert not validate_python_file(str(tmp_path / "missing.py"))


def test_load_python_module(temp_server_file):
    """Test loading a Python module."""
    module = load_python_module(str(temp_server_file))
    assert module is not None
    assert hasattr(module, 'add')
    assert hasattr(module, 'greet')


def test_load_python_module_with_local_imports(tmp_path):
    """Test loading a module that imports from local files.

    This reproduces the issue where load_python_module() would fail
    when the loaded module tried to import from files in the same directory.
    """
    # Create a helper module
    helper_file = tmp_path / "helper.py"
    helper_file.write_text("""
def helper_function(x: int) -> int:
    '''A helper function from a local module.'''
    return x * 3
""")

    # Create a server file that imports from the helper
    server_file = tmp_path / "server_with_import.py"
    server_file.write_text("""
from simply_mcp import tool
from helper import helper_function

@tool()
def use_helper(value: int) -> int:
    '''Tool that uses a helper function.'''
    return helper_function(value)
""")

    # This should work with the fix
    module = load_python_module(str(server_file))
    assert module is not None
    assert hasattr(module, 'use_helper')
    assert hasattr(module, 'helper_function')

    # Verify the helper function works
    assert module.helper_function(5) == 15


def test_load_python_module_not_found():
    """Test loading non-existent module."""
    with pytest.raises(FileNotFoundError):
        load_python_module("/nonexistent/file.py")


def test_load_python_module_invalid_file(tmp_path):
    """Test loading non-Python file."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("test")
    with pytest.raises(ValueError):
        load_python_module(str(txt_file))


def test_load_python_module_invalid_spec(tmp_path):
    """Test loading module when spec.from_file_location returns None."""
    # Create a Python file
    py_file = tmp_path / "test.py"
    py_file.write_text("# test")

    # Mock importlib.util.spec_from_file_location to return None
    with patch('importlib.util.spec_from_file_location', return_value=None):
        with pytest.raises(ImportError, match="Could not load module"):
            load_python_module(str(py_file))


def test_detect_api_style_decorator(temp_server_file):
    """Test API style detection for decorator API."""
    # Reset global server
    from simply_mcp.api.decorators import _global_server
    import simply_mcp.api.decorators as decorators
    decorators._global_server = None

    module = load_python_module(str(temp_server_file))
    api_style, server = detect_api_style(module)

    assert api_style == "decorator"
    assert server is not None
    assert isinstance(server, SimplyMCPServer)

    # Verify components were registered
    stats = server.registry.get_stats()
    assert stats['tools'] >= 1
    assert stats['prompts'] >= 1
    assert stats['resources'] >= 1


def test_detect_api_style_builder(temp_builder_file):
    """Test API style detection for builder API."""
    module = load_python_module(str(temp_builder_file))
    api_style, server = detect_api_style(module)

    assert api_style == "builder"
    assert server is not None
    assert isinstance(server, SimplyMCPServer)


def test_detect_api_style_class_based(tmp_path):
    """Test API style detection for class-based API with @mcp_server decorator."""
    # Create a class-based server file
    server_file = tmp_path / "class_server.py"
    server_file.write_text("""
from simply_mcp import mcp_server, SimplyMCPServer

@mcp_server(name="test", version="1.0.0")
class MyServer:
    def __init__(self):
        pass
""")

    module = load_python_module(str(server_file))
    api_style, server = detect_api_style(module)

    assert api_style == "class"
    assert server is not None
    assert isinstance(server, SimplyMCPServer)


def test_detect_api_style_decorator_with_exception(tmp_path):
    """Test API style detection when get_global_server raises exception."""
    # Create a module with no MCP components
    empty_file = tmp_path / "empty.py"
    empty_file.write_text("# empty file")

    module = load_python_module(str(empty_file))

    # Mock get_global_server to raise an exception
    with patch('simply_mcp.cli.utils.get_global_server', side_effect=RuntimeError("No global server")):
        api_style, server = detect_api_style(module)

        # Should handle exception and return unknown
        assert api_style == "unknown"
        assert server is None


def test_detect_api_style_unknown(tmp_path):
    """Test API style detection when no MCP patterns are found."""
    # Create a module with no MCP components
    empty_file = tmp_path / "plain.py"
    empty_file.write_text("""
def regular_function():
    return "not an MCP server"
""")

    # Reset global server
    import simply_mcp.api.decorators as decorators
    decorators._global_server = None

    module = load_python_module(str(empty_file))
    api_style, server = detect_api_style(module)

    assert api_style == "unknown"
    assert server is None


def test_get_server_instance(temp_server_file):
    """Test getting server instance from module."""
    from simply_mcp.api.decorators import _global_server
    import simply_mcp.api.decorators as decorators
    decorators._global_server = None

    module = load_python_module(str(temp_server_file))
    server = get_server_instance(module)

    assert server is not None
    assert isinstance(server, SimplyMCPServer)


def test_create_components_table():
    """Test creating components table."""
    tools = [{"name": "add", "description": "Add numbers"}]
    prompts = [{"name": "greet", "description": "Greet user"}]
    resources = [{"name": "config", "description": "Config data", "uri": "config://test"}]

    table = create_components_table(tools, prompts, resources)
    assert table is not None
    assert table.title == "MCP Server Components"


def test_format_functions():
    """Test formatting utility functions."""
    # These should not raise exceptions
    format_error("Test error", "Error")
    format_success("Test success", "Success")
    format_info("Test info", "Info")


# Test Main CLI


def test_cli_version(runner):
    """Test CLI version command."""
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help(runner):
    """Test CLI help command."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Simply-MCP" in result.output
    assert "run" in result.output
    assert "config" in result.output
    assert "list" in result.output


# Test Run Command


def test_run_command_help(runner):
    """Test run command help."""
    result = runner.invoke(run, ["--help"])
    assert result.exit_code == 0
    assert "Run an MCP server" in result.output


@patch('simply_mcp.cli.run.load_python_module')
@patch('simply_mcp.cli.run.detect_api_style')
@patch('asyncio.run')
def test_run_command_stdio(mock_asyncio_run, mock_detect, mock_load, runner, tmp_path):
    """Test run command with stdio transport."""
    # Create temp file
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    # Mock module and server
    mock_module = MagicMock()
    mock_load.return_value = mock_module

    # Create proper mock with registry
    mock_server = MagicMock()
    mock_server.config = get_default_config()
    mock_registry = MagicMock()
    mock_registry.get_stats.return_value = {'tools': 1, 'prompts': 0, 'resources': 0, 'total': 1}
    mock_server.registry = mock_registry
    mock_detect.return_value = ("decorator", mock_server)

    # Mock async function to raise KeyboardInterrupt
    mock_asyncio_run.side_effect = KeyboardInterrupt()

    result = runner.invoke(run, [str(server_file)])

    # Should exit gracefully on KeyboardInterrupt
    assert result.exit_code == 0
    mock_load.assert_called_once()
    mock_detect.assert_called_once()


def test_run_command_file_not_found(runner):
    """Test run command with non-existent file."""
    result = runner.invoke(run, ["nonexistent_file_12345.py"])
    # Click returns 2 for usage errors (file not exists)
    assert result.exit_code in [1, 2]


@patch('simply_mcp.cli.run.load_python_module')
@patch('simply_mcp.cli.run.detect_api_style')
def test_run_command_no_server(mock_detect, mock_load, runner, tmp_path):
    """Test run command with no server found."""
    server_file = tmp_path / "empty.py"
    server_file.write_text("# empty")

    mock_module = MagicMock()
    mock_load.return_value = mock_module
    mock_detect.return_value = ("unknown", None)

    result = runner.invoke(run, [str(server_file)])
    assert result.exit_code == 1
    assert "No MCP server found" in result.output


def test_run_command_watch_not_implemented(runner, tmp_path):
    """Test run command with --watch flag."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    result = runner.invoke(run, [str(server_file), "--watch"])
    assert result.exit_code == 1
    assert "not yet implemented" in result.output


# Test Config Commands


def test_config_help(runner):
    """Test config command help."""
    result = runner.invoke(config, ["--help"])
    assert result.exit_code == 0
    assert "configuration" in result.output.lower()


def test_config_init_toml(runner, tmp_path):
    """Test config init command with TOML format."""
    output_file = tmp_path / "test.toml"

    result = runner.invoke(init, ["--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()

    content = output_file.read_text()
    assert "[server]" in content
    assert "[transport]" in content
    assert "[logging]" in content


def test_config_init_json(runner, tmp_path):
    """Test config init command with JSON format."""
    output_file = tmp_path / "test.json"

    result = runner.invoke(init, ["--output", str(output_file), "--format", "json"])
    assert result.exit_code == 0
    assert output_file.exists()

    content = output_file.read_text()
    data = json.loads(content)
    assert "server" in data
    assert "transport" in data


def test_config_init_file_exists(runner, tmp_path):
    """Test config init with existing file."""
    output_file = tmp_path / "test.toml"
    output_file.write_text("existing")

    result = runner.invoke(init, ["--output", str(output_file)])
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_config_init_force_overwrite(runner, tmp_path):
    """Test config init with --force flag."""
    output_file = tmp_path / "test.toml"
    output_file.write_text("existing")

    result = runner.invoke(init, ["--output", str(output_file), "--force"])
    assert result.exit_code == 0
    assert output_file.exists()
    content = output_file.read_text()
    assert "[server]" in content


def test_config_validate_valid(runner, temp_config_file):
    """Test config validate with valid file."""
    result = runner.invoke(validate, [str(temp_config_file)])
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_config_validate_not_found(runner):
    """Test config validate with non-existent file."""
    result = runner.invoke(validate, ["nonexistent_config_12345.toml"])
    # Click returns 2 for usage errors (file not exists)
    assert result.exit_code in [1, 2]


def test_config_show_table(runner, temp_config_file):
    """Test config show command with table format."""
    result = runner.invoke(show, [str(temp_config_file)])
    assert result.exit_code == 0
    assert "Server" in result.output


def test_config_show_json(runner, temp_config_file):
    """Test config show command with JSON format."""
    result = runner.invoke(show, [str(temp_config_file), "--format", "json"])
    assert result.exit_code == 0
    # Just check that JSON-like content is in output
    assert "{" in result.output
    assert "server" in result.output


def test_config_show_toml(runner, temp_config_file):
    """Test config show command with TOML format."""
    result = runner.invoke(show, [str(temp_config_file), "--format", "toml"])
    assert result.exit_code == 0


# Test List Command


def test_list_command_help(runner):
    """Test list command help."""
    result = runner.invoke(list_components, ["--help"])
    assert result.exit_code == 0
    assert "List all components" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_all_components(mock_detect, mock_load, runner, tmp_path):
    """Test list command showing all components."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    # Create mock server with components
    mock_server = MagicMock()
    mock_registry = MagicMock()

    # Create mock objects with attribute access
    mock_tool = MagicMock()
    mock_tool.name = "add"
    mock_tool.description = "Add numbers"

    mock_prompt = MagicMock()
    mock_prompt.name = "greet"
    mock_prompt.description = "Greet user"
    mock_prompt.arguments = []

    mock_resource = MagicMock()
    mock_resource.uri = "config://test"
    mock_resource.name = "config"
    mock_resource.description = "Config"
    mock_resource.mime_type = "application/json"

    mock_registry.list_tools.return_value = [mock_tool]
    mock_registry.list_prompts.return_value = [mock_prompt]
    mock_registry.list_resources.return_value = [mock_resource]
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 0
    assert "add" in result.output
    assert "greet" in result.output
    assert "config" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_filter_tools(mock_detect, mock_load, runner, tmp_path):
    """Test list command with --tools filter."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    # Create mock tool with attribute access
    mock_tool = MagicMock()
    mock_tool.name = "add"
    mock_tool.description = "Add numbers"

    mock_registry.list_tools.return_value = [mock_tool]
    mock_registry.list_prompts.return_value = []
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--tools"])
    assert result.exit_code == 0
    assert "add" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_json_output(mock_detect, mock_load, runner, tmp_path):
    """Test list command with JSON output."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    # Create mock tool with attribute access
    mock_tool = MagicMock()
    mock_tool.name = "add"
    mock_tool.description = "Add numbers"
    mock_tool.input_schema = {}

    mock_registry.list_tools.return_value = [mock_tool]
    mock_registry.list_prompts.return_value = []
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--json"])
    assert result.exit_code == 0
    # Just check JSON-like content in output
    assert "{" in result.output
    assert "tools" in result.output
    assert "add" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_empty_server(mock_detect, mock_load, runner, tmp_path):
    """Test list command with empty server."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()
    mock_registry.list_tools.return_value = []
    mock_registry.list_prompts.return_value = []
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 0
    assert "No components found" in result.output or "0 component" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
def test_list_command_no_server(mock_load, runner, tmp_path):
    """Test list command with no server found."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    with patch('simply_mcp.cli.list_cmd.detect_api_style', return_value=("unknown", None)):
        result = runner.invoke(list_components, [str(server_file)])
        assert result.exit_code == 1
        assert "No MCP server found" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
def test_list_command_import_error(mock_load, runner, tmp_path):
    """Test list command with import error."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_load.side_effect = ImportError("Import failed")

    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 1


@patch('simply_mcp.cli.list_cmd.load_python_module')
def test_list_command_file_not_found_error(mock_load, runner, tmp_path):
    """Test list command with file not found error."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_load.side_effect = FileNotFoundError("File not found")

    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 1
    assert "File Not Found" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
def test_list_command_generic_error(mock_load, runner, tmp_path):
    """Test list command with generic load error."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_load.side_effect = Exception("Unknown error")

    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 1


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_filter_prompts(mock_detect, mock_load, runner, tmp_path):
    """Test list command with --prompts filter."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    mock_prompt = MagicMock()
    mock_prompt.name = "greet"
    mock_prompt.description = "Greet user"
    mock_prompt.arguments = ["name"]

    mock_registry.list_tools.return_value = []
    mock_registry.list_prompts.return_value = [mock_prompt]
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--prompts"])
    assert result.exit_code == 0
    assert "greet" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_filter_resources(mock_detect, mock_load, runner, tmp_path):
    """Test list command with --resources filter."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    mock_resource = MagicMock()
    mock_resource.uri = "file://test.txt"
    mock_resource.name = "test"
    mock_resource.description = "Test resource"
    mock_resource.mime_type = "text/plain"

    mock_registry.list_tools.return_value = []
    mock_registry.list_prompts.return_value = []
    mock_registry.list_resources.return_value = [mock_resource]
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--resources"])
    assert result.exit_code == 0
    assert "test" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_multiple_filters(mock_detect, mock_load, runner, tmp_path):
    """Test list command with multiple filters (tools and prompts)."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    mock_tool = MagicMock()
    mock_tool.name = "add"
    mock_tool.description = "Add numbers"

    mock_prompt = MagicMock()
    mock_prompt.name = "greet"
    mock_prompt.description = "Greet user"
    mock_prompt.arguments = []

    mock_registry.list_tools.return_value = [mock_tool]
    mock_registry.list_prompts.return_value = [mock_prompt]
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--tools", "--prompts"])
    assert result.exit_code == 0
    assert "add" in result.output
    assert "greet" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_json_with_filter(mock_detect, mock_load, runner, tmp_path):
    """Test list command with JSON output and filter."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    mock_prompt = MagicMock()
    mock_prompt.name = "greet"
    mock_prompt.description = "Greet user"
    mock_prompt.arguments = ["name", "greeting"]

    mock_registry.list_tools.return_value = []
    mock_registry.list_prompts.return_value = [mock_prompt]
    mock_registry.list_resources.return_value = []
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--prompts", "--json"])
    assert result.exit_code == 0
    assert "prompts" in result.output
    assert "greet" in result.output


@patch('simply_mcp.cli.list_cmd.load_python_module')
@patch('simply_mcp.cli.list_cmd.detect_api_style')
def test_list_command_json_resources(mock_detect, mock_load, runner, tmp_path):
    """Test list command with JSON output for resources."""
    server_file = tmp_path / "server.py"
    server_file.write_text("# test")

    mock_module = MagicMock()
    mock_load.return_value = mock_module

    mock_server = MagicMock()
    mock_registry = MagicMock()

    mock_resource = MagicMock()
    mock_resource.uri = "file://test.txt"
    mock_resource.name = "test"
    mock_resource.description = "Test resource"
    mock_resource.mime_type = "text/plain"

    mock_registry.list_tools.return_value = []
    mock_registry.list_prompts.return_value = []
    mock_registry.list_resources.return_value = [mock_resource]
    mock_server.registry = mock_registry

    mock_detect.return_value = ("decorator", mock_server)

    result = runner.invoke(list_components, [str(server_file), "--resources", "--json"])
    assert result.exit_code == 0
    assert "resources" in result.output
    assert "test" in result.output


# Integration Tests


def test_full_workflow_decorator_api(runner, tmp_path):
    """Test full workflow with decorator API."""
    # Create server file
    server_file = tmp_path / "server.py"
    server_file.write_text("""
from simply_mcp import tool

@tool()
def test_tool(x: int) -> int:
    '''Test tool.'''
    return x * 2
""")

    # Reset global server
    import simply_mcp.api.decorators as decorators
    decorators._global_server = None

    # List components
    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 0


def test_full_workflow_builder_api(runner, tmp_path):
    """Test full workflow with builder API."""
    # Create server file
    server_file = tmp_path / "server.py"
    server_file.write_text("""
from simply_mcp import BuildMCPServer

mcp = BuildMCPServer(name="test", version="1.0.0")

@mcp.tool()
def test_tool(x: int) -> int:
    '''Test tool.'''
    return x * 2
""")

    # List components
    result = runner.invoke(list_components, [str(server_file)])
    assert result.exit_code == 0


__all__ = [
    "test_validate_python_file",
    "test_load_python_module",
    "test_load_python_module_with_local_imports",
    "test_detect_api_style_decorator",
    "test_detect_api_style_builder",
    "test_cli_version",
    "test_run_command_stdio",
    "test_config_init_toml",
    "test_list_command_all_components",
]
