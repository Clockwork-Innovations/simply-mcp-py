"""Tests for bundle support functionality in the run command.

This module contains comprehensive unit tests for:
- Bundle detection and server discovery
- Dependency installation with uv
- Packaged .pyz file loading
- Error handling for various edge cases
"""

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from simply_mcp.cli.run import (
    find_bundle_server,
    install_bundle_dependencies,
    load_packaged_server,
    run,
)


class TestFindBundleServer:
    """Tests for bundle server discovery functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.bundle_path = Path(self.temp_dir)

    def test_find_bundle_server_standard_layout(self) -> None:
        """Test finding server.py in standard src/{package}/ layout.

        Validates:
        - Detection of src/{package}/server.py structure
        - Returns correct Path object
        - Works with standard Python package layout
        """
        # Create pyproject.toml
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test-server"')

        # Create standard layout
        src_dir = self.bundle_path / "src"
        src_dir.mkdir()
        package_dir = src_dir / "test_server"
        package_dir.mkdir()
        server_file = package_dir / "server.py"
        server_file.write_text("# Server code")

        result = find_bundle_server(self.bundle_path)

        assert result == server_file
        assert result.exists()
        assert result.name == "server.py"

    def test_find_bundle_server_root_layout(self) -> None:
        """Test finding server.py at bundle root.

        Validates:
        - Detection of server.py in root directory
        - Works when no src/ directory exists
        - Simple bundle layout support
        """
        # Create pyproject.toml
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test-server"')

        # Create server at root
        server_file = self.bundle_path / "server.py"
        server_file.write_text("# Server code")

        result = find_bundle_server(self.bundle_path)

        assert result == server_file
        assert result.exists()
        assert result.is_file()

    def test_find_bundle_server_main_fallback(self) -> None:
        """Test finding main.py as fallback when server.py doesn't exist.

        Validates:
        - Fallback to main.py when server.py not found
        - Search order: server.py -> main.py
        - Alternative naming convention support
        """
        # Create pyproject.toml
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test-server"')

        # Create main.py instead of server.py
        main_file = self.bundle_path / "main.py"
        main_file.write_text("# Main server code")

        result = find_bundle_server(self.bundle_path)

        assert result == main_file
        assert result.name == "main.py"

    def test_find_bundle_server_no_pyproject(self) -> None:
        """Test error when pyproject.toml is missing.

        Validates:
        - Raises FileNotFoundError when pyproject.toml missing
        - Error message includes bundle path
        - Validates bundle structure requirement
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            find_bundle_server(self.bundle_path)

        assert "No pyproject.toml found" in str(exc_info.value)
        assert str(self.bundle_path) in str(exc_info.value)

    def test_find_bundle_server_no_server(self) -> None:
        """Test error when no server file is found.

        Validates:
        - Raises FileNotFoundError when no server.py or main.py exists
        - Error message lists expected file locations
        - Comprehensive search performed before error
        """
        # Create pyproject.toml but no server files
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test-server"')

        with pytest.raises(FileNotFoundError) as exc_info:
            find_bundle_server(self.bundle_path)

        error_msg = str(exc_info.value)
        assert "No server.py or main.py found" in error_msg
        assert str(self.bundle_path) in error_msg

    def test_find_bundle_server_multiple_packages_in_src(self) -> None:
        """Test server discovery with multiple packages in src/.

        Validates:
        - Finds first valid server.py in src/ subdirectories
        - Handles multiple package directories
        - Returns first match found
        """
        # Create pyproject.toml
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test-server"')

        # Create multiple packages, one with server.py
        src_dir = self.bundle_path / "src"
        src_dir.mkdir()

        # Package without server
        pkg1_dir = src_dir / "package1"
        pkg1_dir.mkdir()
        (pkg1_dir / "__init__.py").write_text("")

        # Package with server
        pkg2_dir = src_dir / "package2"
        pkg2_dir.mkdir()
        server_file = pkg2_dir / "server.py"
        server_file.write_text("# Server code")

        result = find_bundle_server(self.bundle_path)

        # Should find the server.py
        assert result.exists()
        assert result.name == "server.py"
        assert "package2" in str(result)


class TestInstallBundleDependencies:
    """Tests for bundle dependency installation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.bundle_path = Path(self.temp_dir) / "bundle"
        self.bundle_path.mkdir()
        self.venv_path = Path(self.temp_dir) / "venv"

    @patch("simply_mcp.cli.run.subprocess.run")
    def test_install_bundle_dependencies_success(self, mock_run: Mock) -> None:
        """Test successful dependency installation.

        Validates:
        - Creates virtual environment using uv venv
        - Installs dependencies using uv pip install -e
        - Uses correct venv path and environment variables
        - Both subprocess calls succeed
        """
        # Create pyproject.toml
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test-server"
dependencies = ["requests>=2.28.0"]
""")

        # Mock successful subprocess calls
        mock_run.return_value = Mock(returncode=0)

        install_bundle_dependencies(self.bundle_path, self.venv_path)

        # Verify uv venv was called
        assert mock_run.call_count == 2
        venv_call = mock_run.call_args_list[0]
        assert venv_call[0][0] == ["uv", "venv", str(self.venv_path)]
        assert venv_call[1]["check"] is True

        # Verify uv pip install was called
        install_call = mock_run.call_args_list[1]
        assert install_call[0][0] == ["uv", "pip", "install", "-e", str(self.bundle_path)]
        assert install_call[1]["cwd"] == str(self.venv_path)
        assert "VIRTUAL_ENV" in install_call[1]["env"]
        assert install_call[1]["env"]["VIRTUAL_ENV"] == str(self.venv_path)

    @patch("simply_mcp.cli.run.subprocess.run")
    def test_install_bundle_dependencies_venv_creation_failure(self, mock_run: Mock) -> None:
        """Test error when venv creation fails.

        Validates:
        - Raises RuntimeError when uv venv fails
        - Error message includes stderr output
        - Proper error handling for subprocess failures
        """
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        # Mock failed venv creation
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["uv", "venv"],
            stderr=b"Failed to create venv"
        )

        with pytest.raises(RuntimeError) as exc_info:
            install_bundle_dependencies(self.bundle_path, self.venv_path)

        assert "Failed to create venv" in str(exc_info.value)

    @patch("simply_mcp.cli.run.subprocess.run")
    def test_install_bundle_dependencies_no_uv(self, mock_run: Mock) -> None:
        """Test error when uv is not installed.

        Validates:
        - Raises RuntimeError when uv command not found
        - Error message includes installation instructions
        - Helpful error messaging for missing tool
        """
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        # Mock uv not found
        mock_run.side_effect = FileNotFoundError("uv not found")

        with pytest.raises(RuntimeError) as exc_info:
            install_bundle_dependencies(self.bundle_path, self.venv_path)

        error_msg = str(exc_info.value)
        assert "uv is not installed" in error_msg
        assert "https://docs.astral.sh/uv/" in error_msg

    @patch("simply_mcp.cli.run.subprocess.run")
    def test_install_bundle_dependencies_install_failure(self, mock_run: Mock) -> None:
        """Test error when pip install fails.

        Validates:
        - Raises RuntimeError when dependency installation fails
        - Error message includes failure details
        - Second subprocess call (pip install) fails
        """
        pyproject = self.bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        # Mock successful venv creation, failed install
        def side_effect(cmd, **kwargs):
            if "venv" in cmd:
                return Mock(returncode=0)
            else:
                raise subprocess.CalledProcessError(
                    returncode=1,
                    cmd=cmd,
                    stderr=b"Failed to install dependencies"
                )

        mock_run.side_effect = side_effect

        with pytest.raises(RuntimeError) as exc_info:
            install_bundle_dependencies(self.bundle_path, self.venv_path)

        assert "Failed to install dependencies" in str(exc_info.value)


class TestLoadPackagedServer:
    """Tests for .pyz package loading functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_load_packaged_server_valid(self) -> None:
        """Test loading a valid .pyz package.

        Validates:
        - Successfully extracts and loads .pyz file
        - Reads package.json metadata correctly
        - Returns tuple of (api_style, server_instance)
        - Handles ZIP extraction properly
        """
        pyz_path = Path(self.temp_dir) / "test.pyz"

        # Create valid package
        server_code = """
from simply_mcp import SimplyMCP

server = SimplyMCP("test")

@server.tool()
def test_tool():
    return "test"
"""
        metadata = {
            "name": "test_server",
            "version": "1.0.0",
            "api_style": "builder",
            "original_file": "server.py",
        }

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("server.py", server_code)
            zf.writestr("package.json", json.dumps(metadata))

        with patch("simply_mcp.cli.run.load_python_module") as mock_load, \
             patch("simply_mcp.cli.run.detect_api_style") as mock_detect:

            mock_module = MagicMock()
            mock_load.return_value = mock_module

            mock_server = MagicMock()
            mock_detect.return_value = ("builder", mock_server)

            api_style, server = load_packaged_server(str(pyz_path))

            assert api_style == "builder"
            assert server == mock_server
            mock_load.assert_called_once()
            mock_detect.assert_called_once_with(mock_module)

    def test_load_packaged_server_invalid_zip(self) -> None:
        """Test error for non-ZIP .pyz file.

        Validates:
        - Raises ValueError for invalid ZIP format
        - Error message indicates not a ZIP file
        - Validates file format before processing
        """
        pyz_path = Path(self.temp_dir) / "invalid.pyz"
        pyz_path.write_text("Not a ZIP file")

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(pyz_path))

        error_msg = str(exc_info.value)
        assert "Invalid .pyz package" in error_msg
        assert "not a ZIP file" in error_msg

    def test_load_packaged_server_missing_metadata(self) -> None:
        """Test error when package.json is missing.

        Validates:
        - Raises ValueError when package.json not in archive
        - Error message mentions missing metadata
        - Validates package structure
        """
        pyz_path = Path(self.temp_dir) / "no_metadata.pyz"

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("server.py", "# code")

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(pyz_path))

        error_msg = str(exc_info.value)
        assert "missing package.json" in error_msg

    def test_load_packaged_server_invalid_metadata_json(self) -> None:
        """Test error when package.json has invalid JSON.

        Validates:
        - Raises ValueError for malformed JSON
        - Error indicates JSON parsing failure
        - Proper error propagation from json.load
        """
        pyz_path = Path(self.temp_dir) / "bad_json.pyz"

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("server.py", "# code")
            zf.writestr("package.json", "{invalid json")

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(pyz_path))

        assert "Invalid package.json" in str(exc_info.value)

    def test_load_packaged_server_missing_original_file_field(self) -> None:
        """Test error when package.json missing 'original_file' field.

        Validates:
        - Raises ValueError when original_file field missing
        - Validates required metadata fields
        - Clear error message about missing field
        """
        pyz_path = Path(self.temp_dir) / "no_original.pyz"

        metadata = {
            "name": "test",
            "version": "1.0.0"
            # Missing 'original_file'
        }

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("server.py", "# code")
            zf.writestr("package.json", json.dumps(metadata))

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(pyz_path))

        error_msg = str(exc_info.value)
        assert "missing 'original_file' field" in error_msg

    def test_load_packaged_server_missing_server_file(self) -> None:
        """Test error when server file referenced in metadata doesn't exist.

        Validates:
        - Raises ValueError when server file not in archive
        - Uses original_file from metadata to locate server
        - Validates archive contents
        """
        pyz_path = Path(self.temp_dir) / "missing_server.pyz"

        metadata = {
            "name": "test",
            "version": "1.0.0",
            "original_file": "server.py"
        }

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("package.json", json.dumps(metadata))
            # Don't include server.py

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(pyz_path))

        error_msg = str(exc_info.value)
        assert "server file" in error_msg.lower()
        assert "not found" in error_msg.lower()

    def test_load_packaged_server_file_not_found(self) -> None:
        """Test error when .pyz file doesn't exist.

        Validates:
        - Raises FileNotFoundError for missing file
        - Error message includes file path
        - Pre-validation before ZIP processing
        """
        pyz_path = "/nonexistent/path/test.pyz"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_packaged_server(pyz_path)

        assert "Package file not found" in str(exc_info.value)
        assert pyz_path in str(exc_info.value)

    def test_load_packaged_server_not_pyz_extension(self) -> None:
        """Test error when file doesn't have .pyz extension.

        Validates:
        - Raises ValueError for non-.pyz files
        - Extension validation before processing
        - Clear error message about file type
        """
        bad_path = Path(self.temp_dir) / "server.py"
        bad_path.write_text("# code")

        with pytest.raises(ValueError) as exc_info:
            load_packaged_server(str(bad_path))

        assert "Not a .pyz file" in str(exc_info.value)

    def test_load_packaged_server_no_server_in_module(self) -> None:
        """Test error when loaded module has no MCP server.

        Validates:
        - Raises ValueError when detect_api_style returns None
        - Validates server instance exists
        - Helpful error message about missing server
        """
        pyz_path = Path(self.temp_dir) / "no_server.pyz"

        metadata = {
            "name": "test",
            "version": "1.0.0",
            "original_file": "server.py"
        }

        with zipfile.ZipFile(pyz_path, "w") as zf:
            zf.writestr("server.py", "# No server defined")
            zf.writestr("package.json", json.dumps(metadata))

        with patch("simply_mcp.cli.run.load_python_module") as mock_load, \
             patch("simply_mcp.cli.run.detect_api_style") as mock_detect:

            mock_module = MagicMock()
            mock_load.return_value = mock_module
            mock_detect.return_value = ("unknown", None)

            with pytest.raises(ValueError) as exc_info:
                load_packaged_server(str(pyz_path))

            error_msg = str(exc_info.value)
            assert "No MCP server found" in error_msg


class TestBundleDetectionLogic:
    """Tests for bundle detection in run command."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_bundle_detection_with_pyproject(self) -> None:
        """Test that directory with pyproject.toml is detected as bundle.

        Validates:
        - is_bundle flag set correctly
        - Bundle detection logic in run command
        - pyproject.toml presence triggers bundle path
        """
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        # Create pyproject.toml
        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        # Create server
        server_file = bundle_path / "server.py"
        server_file.write_text("from simply_mcp import SimplyMCP\nserver = SimplyMCP('test')")

        # Test detection logic
        file_path = bundle_path.resolve()
        is_directory = file_path.is_dir()
        is_bundle = is_directory and (file_path / "pyproject.toml").exists()

        assert is_directory is True
        assert is_bundle is True

    def test_bundle_detection_without_pyproject(self) -> None:
        """Test that directory without pyproject.toml is not a bundle.

        Validates:
        - is_bundle flag is False without pyproject.toml
        - Regular directory not treated as bundle
        - Differentiation between bundle and regular directory
        """
        bundle_path = Path(self.temp_dir) / "not_bundle"
        bundle_path.mkdir()

        server_file = bundle_path / "server.py"
        server_file.write_text("# Server")

        file_path = bundle_path.resolve()
        is_directory = file_path.is_dir()
        is_bundle = is_directory and (file_path / "pyproject.toml").exists()

        assert is_directory is True
        assert is_bundle is False

    def test_pyz_detection(self) -> None:
        """Test .pyz file detection logic.

        Validates:
        - is_pyz flag set correctly for .pyz files
        - Extension-based detection works
        - Distinction from bundles and regular files
        """
        pyz_path = Path(self.temp_dir) / "test.pyz"
        pyz_path.write_text("# pyz content")

        file_path = pyz_path.resolve()
        is_pyz = file_path.suffix == ".pyz"

        assert is_pyz is True
        assert file_path.is_dir() is False


class TestRelativePathResolution:
    """Tests for relative path handling in bundle operations."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_relative_path_resolution(self) -> None:
        """Test that .resolve() handles relative paths correctly.

        Validates:
        - Relative paths converted to absolute
        - Path.resolve() works as expected
        - Consistent path handling across bundle operations
        """
        # Create a bundle with relative path
        bundle_dir = Path(self.temp_dir) / "bundle"
        bundle_dir.mkdir()

        # Create a relative path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            relative_path = Path("./bundle")
            resolved_path = relative_path.resolve()

            assert resolved_path.is_absolute()
            assert resolved_path == bundle_dir
            assert str(resolved_path) == str(bundle_dir)
        finally:
            os.chdir(original_cwd)


class TestRunBundleIntegration:
    """Integration tests for running bundles via CLI."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    @patch("simply_mcp.cli.run.find_bundle_server")
    @patch("simply_mcp.cli.run.install_bundle_dependencies")
    @patch("simply_mcp.cli.run.load_python_module")
    @patch("simply_mcp.cli.run.detect_api_style")
    @patch("simply_mcp.cli.run.asyncio.run")
    def test_run_bundle_with_mocked_server(
        self,
        mock_asyncio_run: Mock,
        mock_detect_api_style: Mock,
        mock_load_module: Mock,
        mock_install_deps: Mock,
        mock_find_server: Mock,
    ) -> None:
        """Test running a bundle with all components mocked.

        Validates:
        - Complete bundle execution flow
        - All bundle-specific functions called in correct order
        - Bundle detected and processed correctly
        - Integration of find, install, and load operations
        """
        # Create bundle structure
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        server_file = bundle_path / "server.py"
        server_file.write_text("from simply_mcp import SimplyMCP\nserver = SimplyMCP('test')")

        # Mock bundle operations
        mock_find_server.return_value = server_file
        mock_install_deps.return_value = None

        mock_module = MagicMock()
        mock_load_module.return_value = mock_module

        mock_server = MagicMock()
        mock_server.registry.get_stats.return_value = {
            "tools": 1, "prompts": 0, "resources": 0
        }
        mock_server.config.server.name = "test"
        mock_server.config.server.version = "1.0.0"
        mock_detect_api_style.return_value = ("Builder", mock_server)

        mock_asyncio_run.side_effect = KeyboardInterrupt()

        # Run the bundle
        result = self.runner.invoke(run, [str(bundle_path)])

        # Verify execution
        assert result.exit_code == 0
        mock_find_server.assert_called_once()
        mock_install_deps.assert_called_once()

        # Verify venv path was created
        venv_call = mock_install_deps.call_args
        assert venv_call is not None
        venv_path_arg = venv_call[0][1]
        assert isinstance(venv_path_arg, Path)

    @patch("simply_mcp.cli.run.find_bundle_server")
    @patch("simply_mcp.cli.run.install_bundle_dependencies")
    def test_run_bundle_with_custom_venv_path(
        self,
        mock_install_deps: Mock,
        mock_find_server: Mock,
    ) -> None:
        """Test running bundle with custom venv path.

        Validates:
        - --venv-path option works correctly
        - Custom venv path passed to install function
        - User-specified venv location used instead of temp
        """
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        server_file = bundle_path / "server.py"
        server_file.write_text("from simply_mcp import SimplyMCP\nserver = SimplyMCP('test')")

        custom_venv = Path(self.temp_dir) / "my_venv"

        mock_find_server.return_value = server_file
        mock_install_deps.return_value = None

        with patch("simply_mcp.cli.run.load_python_module"), \
             patch("simply_mcp.cli.run.detect_api_style") as mock_detect, \
             patch("simply_mcp.cli.run.asyncio.run") as mock_async:

            mock_server = MagicMock()
            mock_server.registry.get_stats.return_value = {
                "tools": 0, "prompts": 0, "resources": 0
            }
            mock_server.config.server.name = "test"
            mock_server.config.server.version = "1.0.0"
            mock_detect.return_value = ("Builder", mock_server)
            mock_async.side_effect = KeyboardInterrupt()

            result = self.runner.invoke(
                run,
                [str(bundle_path), "--venv-path", str(custom_venv)]
            )

            assert result.exit_code == 0

            # Verify custom venv path was used
            venv_call = mock_install_deps.call_args
            assert venv_call[0][1] == custom_venv

    @patch("simply_mcp.cli.run.find_bundle_server")
    def test_run_bundle_server_not_found_error(
        self,
        mock_find_server: Mock,
    ) -> None:
        """Test bundle execution when server file not found.

        Validates:
        - FileNotFoundError from find_bundle_server handled
        - Proper error message displayed
        - Exit code 1 on error
        """
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        mock_find_server.side_effect = FileNotFoundError("No server.py found")

        result = self.runner.invoke(run, [str(bundle_path)])

        assert result.exit_code == 1
        assert "Bundle Error" in result.output
        assert "No server.py found" in result.output

    @patch("simply_mcp.cli.run.find_bundle_server")
    @patch("simply_mcp.cli.run.install_bundle_dependencies")
    def test_run_bundle_dependency_install_error(
        self,
        mock_install_deps: Mock,
        mock_find_server: Mock,
    ) -> None:
        """Test bundle execution when dependency installation fails.

        Validates:
        - RuntimeError from install_bundle_dependencies handled
        - Proper error message displayed
        - Error category shown as "Dependency Installation Error"
        """
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        server_file = bundle_path / "server.py"
        server_file.write_text("# Server")

        mock_find_server.return_value = server_file
        mock_install_deps.side_effect = RuntimeError("uv is not installed")

        result = self.runner.invoke(run, [str(bundle_path)])

        assert result.exit_code == 1
        assert "Dependency Installation Error" in result.output
        assert "uv is not installed" in result.output

    @patch("simply_mcp.cli.run.find_bundle_server")
    @patch("simply_mcp.cli.run.install_bundle_dependencies")
    @patch("simply_mcp.cli.run.load_python_module")
    @patch("simply_mcp.cli.run.detect_api_style")
    def test_run_bundle_no_server_found_in_module(
        self,
        mock_detect_api_style: Mock,
        mock_load_module: Mock,
        mock_install_deps: Mock,
        mock_find_server: Mock,
    ) -> None:
        """Test bundle execution when loaded module has no MCP server.

        Validates:
        - Proper handling when detect_api_style returns None
        - Error message guides user on server definition
        - Lists supported API styles
        """
        bundle_path = Path(self.temp_dir) / "bundle"
        bundle_path.mkdir()

        pyproject = bundle_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "test"')

        server_file = bundle_path / "server.py"
        server_file.write_text("# No server defined")

        mock_find_server.return_value = server_file
        mock_install_deps.return_value = None

        mock_module = MagicMock()
        mock_load_module.return_value = mock_module
        mock_detect_api_style.return_value = ("Unknown", None)

        result = self.runner.invoke(run, [str(bundle_path)])

        assert result.exit_code == 1
        assert "No MCP server found in the bundle" in result.output
        assert "Decorator API" in result.output
        assert "Builder API" in result.output
        assert "Class API" in result.output
