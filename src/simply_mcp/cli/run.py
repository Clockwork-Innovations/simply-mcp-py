"""Run command for Simply-MCP CLI.

This module implements the 'run' command which loads and executes
MCP servers from Python files, packaged .pyz files, or server bundles.
It supports auto-detection of API styles, multiple transport types,
and automatic dependency installation from bundle pyproject.toml files.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel

from simply_mcp.cli.utils import (
    console,
    detect_api_style,
    format_error,
    format_info,
    format_success,
    load_python_module,
)
from simply_mcp.core.config import load_config
from simply_mcp.core.errors import SimplyMCPError


def find_bundle_server(bundle_path: Path) -> Path:
    """Find the server entry point in a bundle directory.

    Searches for:
    1. src/{package_name}/server.py (standard layout)
    2. {package_name}.py (simple layout)
    3. server.py (root layout)

    Args:
        bundle_path: Path to the bundle directory

    Returns:
        Path to the server file

    Raises:
        FileNotFoundError: If no server file is found
    """
    # Check for pyproject.toml
    pyproject_path = bundle_path / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"No pyproject.toml found in {bundle_path}")

    # Try standard src/ layout first
    src_dir = bundle_path / "src"
    if src_dir.exists():
        for item in src_dir.iterdir():
            if item.is_dir():
                server_py = item / "server.py"
                if server_py.exists():
                    return server_py

    # Try simple layout in root
    for pattern in ["server.py", "main.py"]:
        simple_server = bundle_path / pattern
        if simple_server.exists():
            return simple_server

    raise FileNotFoundError(
        f"No server.py or main.py found in {bundle_path} or src/ subdirectories"
    )


def install_bundle_dependencies(bundle_path: Path, venv_path: Path) -> None:
    """Install bundle dependencies using uv into a virtual environment.

    Args:
        bundle_path: Path to the bundle directory (with pyproject.toml)
        venv_path: Path to create/use as virtual environment

    Raises:
        RuntimeError: If dependency installation fails
    """
    format_info(f"Creating virtual environment at: {venv_path}")

    # Create venv using uv
    try:
        subprocess.run(
            ["uv", "venv", str(venv_path)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to create venv: {e.stderr.decode()}") from e
    except FileNotFoundError as e:
        raise RuntimeError("uv is not installed. Please install uv: https://docs.astral.sh/uv/") from e

    format_info("Installing dependencies from pyproject.toml...")

    # Install dependencies using uv
    try:
        subprocess.run(
            ["uv", "pip", "install", "-e", str(bundle_path)],
            check=True,
            cwd=str(venv_path),
            env={**os.environ, "VIRTUAL_ENV": str(venv_path)},
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to install dependencies: {e.stderr.decode()}") from e

    format_success("Dependencies installed successfully")


def load_packaged_server(pyz_path: str) -> tuple[str, Any]:
    """Load an MCP server from a .pyz package file.

    Args:
        pyz_path: Path to .pyz package file

    Returns:
        Tuple of (api_style, server_instance)

    Raises:
        ValueError: If package is invalid or malformed
        FileNotFoundError: If package file doesn't exist
    """
    pyz_file = Path(pyz_path)

    if not pyz_file.exists():
        raise FileNotFoundError(f"Package file not found: {pyz_path}")

    if not pyz_file.suffix == ".pyz":
        raise ValueError(f"Not a .pyz file: {pyz_path}")

    # Verify it's a valid ZIP file
    if not zipfile.is_zipfile(pyz_file):
        raise ValueError(f"Invalid .pyz package (not a ZIP file): {pyz_path}")

    # Extract to temporary directory
    temp_dir = tempfile.mkdtemp(prefix="simply_mcp_")
    temp_path = Path(temp_dir)

    try:
        # Extract the package
        with zipfile.ZipFile(pyz_file, "r") as zf:
            zf.extractall(temp_path)

        # Load package.json metadata
        package_json = temp_path / "package.json"
        if not package_json.exists():
            raise ValueError(
                "Invalid .pyz package: missing package.json metadata.\n"
                "This may not be a valid Simply-MCP package."
            )

        try:
            with package_json.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid package.json: {e}") from e

        # Extract metadata
        original_file = metadata.get("original_file")
        if not original_file:
            raise ValueError("Invalid package.json: missing 'original_file' field")

        # Determine server module name (without .py extension)
        server_module_name = Path(original_file).stem
        server_file = temp_path / f"{server_module_name}.py"

        if not server_file.exists():
            raise ValueError(
                f"Invalid .pyz package: server file '{server_module_name}.py' not found"
            )

        # Load the server module from extracted location
        try:
            module = load_python_module(str(server_file))
        except Exception as e:
            raise ValueError(f"Failed to load server module: {e}") from e

        # Detect API style and get server instance
        api_style, server = detect_api_style(module)

        if server is None:
            raise ValueError(
                "No MCP server found in the packaged file.\n"
                "The package may be corrupted or invalid."
            )

        return (api_style, server)

    except Exception:
        # Clean up temp directory on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


@click.command()
@click.argument("server_file", type=click.Path(exists=True))
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http", "sse"], case_sensitive=False),
    default="stdio",
    help="Transport type to use (default: stdio)",
)
@click.option(
    "--port",
    type=int,
    default=3000,
    help="Port for network transports (default: 3000)",
)
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host for network transports (default: 0.0.0.0)",
)
@click.option(
    "--cors/--no-cors",
    default=True,
    help="Enable/disable CORS for network transports (default: enabled)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to configuration file",
)
@click.option(
    "--watch",
    is_flag=True,
    help="Enable auto-reload on file changes (Phase 4)",
)
@click.option(
    "--venv-path",
    type=click.Path(),
    default=None,
    help="Path for virtual environment (used for bundles with dependencies)",
)
def run(
    server_file: str,
    transport: str,
    port: int,
    host: str,
    cors: bool,
    config: str | None,
    watch: bool,
    venv_path: str | None,
) -> None:
    """Run an MCP server from a Python file, bundle, or .pyz package.

    This command loads and executes MCP servers from multiple sources:
    - Python file: simply-mcp run server.py
    - Bundle directory with pyproject.toml: simply-mcp run ./my-bundle/
    - Packaged .pyz file: simply-mcp run package.pyz

    For bundles, dependencies are automatically installed using uv into
    a virtual environment before running the server.

    Examples:

        \b
        # Run Python file with stdio transport (default)
        simply-mcp run server.py

        \b
        # Run a bundle (auto-installs dependencies)
        simply-mcp run ./gemini-server/

        \b
        # Run packaged .pyz file
        simply-mcp run package.pyz

        \b
        # Run bundle with custom venv location
        simply-mcp run ./gemini-server/ --venv-path ./my-venv

        \b
        # Run with HTTP transport
        simply-mcp run server.py --transport http --port 8080

        \b
        # Run with custom config
        simply-mcp run server.py --config myconfig.toml
    """
    if watch:
        format_error(
            "Auto-reload (--watch) is not yet implemented. This feature is planned for Phase 4.",
            "Not Implemented"
        )
        sys.exit(1)

    try:
        # Display startup info
        # Build transport info string
        transport_info = f"Transport: [yellow]{transport}[/yellow]"
        if transport in ["http", "sse"]:
            transport_info += f"\nHost: [yellow]{host}[/yellow]"
            transport_info += f"\nPort: [yellow]{port}[/yellow]"
            transport_info += f"\nCORS: [yellow]{'enabled' if cors else 'disabled'}[/yellow]"

        console.print(Panel(
            f"[bold cyan]Starting Simply-MCP Server[/bold cyan]\n\n"
            f"File: [green]{server_file}[/green]\n"
            f"{transport_info}",
            title="[bold blue]Simply-MCP[/bold blue]",
        ))

        # Load configuration if provided
        server_config = None
        if config:
            try:
                server_config = load_config(config)
                format_info(f"Loaded configuration from: {config}")
            except Exception as e:
                format_error(f"Failed to load configuration: {e}", "Configuration Error")
                sys.exit(1)

        # Update port if provided
        if port and server_config:
            server_config.transport.port = port

        # Detect file type and load appropriately
        file_path = Path(server_file).resolve()
        is_directory = file_path.is_dir()
        is_pyz = file_path.suffix == ".pyz"
        is_bundle = is_directory and (file_path / "pyproject.toml").exists()

        if is_bundle:
            # Handle bundle (directory with pyproject.toml)
            console.print("[dim]Loading server bundle...[/dim]")
            try:
                # Find server entry point
                server_entry = find_bundle_server(file_path)
                format_info(f"Found server: {server_entry.relative_to(file_path)}")

                # Install dependencies if needed
                if venv_path is None:
                    venv_path = str(tempfile.mkdtemp(prefix="simply_mcp_venv_"))

                venv_path_obj = Path(venv_path)
                install_bundle_dependencies(file_path, venv_path_obj)

                # Load server from the bundle
                module = load_python_module(str(server_entry))
                api_style, server = detect_api_style(module)

                if server is None:
                    format_error(
                        "No MCP server found in the bundle.\n\n"
                        "Make sure your server file uses one of:\n"
                        "  - Decorator API: @tool(), @prompt(), @resource()\n"
                        "  - Builder API: SimplyMCP(...)\n"
                        "  - Class API: @mcp_server class",
                        "No Server Found"
                    )
                    sys.exit(1)

            except FileNotFoundError as e:
                format_error(str(e), "Bundle Error")
                sys.exit(1)
            except RuntimeError as e:
                format_error(str(e), "Dependency Installation Error")
                sys.exit(1)
            except Exception as e:
                format_error(f"Error loading bundle: {e}", "Load Error")
                sys.exit(1)

        elif is_pyz:
            # Load from .pyz package
            console.print("[dim]Loading packaged server...[/dim]")
            try:
                api_style, server = load_packaged_server(server_file)
            except FileNotFoundError as e:
                format_error(str(e), "File Not Found")
                sys.exit(1)
            except ValueError as e:
                format_error(str(e), "Invalid Package")
                sys.exit(1)
            except Exception as e:
                format_error(f"Error loading package: {e}", "Load Error")
                sys.exit(1)
        else:
            # Load from Python source file
            console.print("[dim]Loading server module...[/dim]")
            try:
                module = load_python_module(server_file)
            except FileNotFoundError as e:
                format_error(str(e), "File Not Found")
                sys.exit(1)
            except ImportError as e:
                format_error(f"Failed to import module: {e}", "Import Error")
                sys.exit(1)
            except Exception as e:
                format_error(f"Error loading module: {e}", "Load Error")
                sys.exit(1)

            # Detect API style and get server instance
            console.print("[dim]Detecting API style...[/dim]")
            api_style, server = detect_api_style(module)

            if server is None:
                format_error(
                    "No MCP server found in the file.\n\n"
                    "Make sure your file uses one of:\n"
                    "  - Decorator API: @tool(), @prompt(), @resource()\n"
                    "  - Builder API: SimplyMCP(...)\n"
                    "  - Class API: @mcp_server class",
                    "No Server Found"
                )
                sys.exit(1)

        format_success(f"Detected {api_style} API style")

        # Display server info
        stats = server.registry.get_stats()
        console.print(Panel(
            f"[bold]Server:[/bold] [cyan]{server.config.server.name}[/cyan]\n"
            f"[bold]Version:[/bold] [cyan]{server.config.server.version}[/cyan]\n"
            f"[bold]Components:[/bold] [green]{stats['tools']} tools, "
            f"{stats['prompts']} prompts, {stats['resources']} resources[/green]",
            title="[bold green]Server Info[/bold green]",
        ))

        # Initialize server
        console.print("[dim]Initializing server...[/dim]")

        async def run_server() -> None:
            """Run the server asynchronously."""
            try:
                await server.initialize()
                format_success("Server initialized successfully")

                # Prepare running message
                if transport == "stdio":
                    running_msg = (
                        f"[bold green]Server is running on {transport} transport[/bold green]\n\n"
                        f"Press [bold]Ctrl+C[/bold] to stop the server."
                    )
                else:
                    running_msg = (
                        f"[bold green]Server is running on {transport} transport[/bold green]\n\n"
                        f"URL: [cyan]http://{host}:{port}[/cyan]\n"
                        f"Endpoints:\n"
                        f"  - [cyan]http://{host}:{port}/[/cyan] (info)\n"
                        f"  - [cyan]http://{host}:{port}/health[/cyan] (health check)\n"
                    )
                    if transport == "http":
                        running_msg += f"  - [cyan]http://{host}:{port}/mcp[/cyan] (JSON-RPC)\n"
                    elif transport == "sse":
                        running_msg += (
                            f"  - [cyan]http://{host}:{port}/sse[/cyan] (SSE stream)\n"
                            f"  - [cyan]http://{host}:{port}/mcp[/cyan] (JSON-RPC)\n"
                        )
                    running_msg += "\nPress [bold]Ctrl+C[/bold] to stop the server."

                console.print(Panel(
                    running_msg,
                    title="[bold cyan]Running[/bold cyan]",
                ))

                # Run with specified transport
                if transport == "stdio":
                    await server.run_stdio()
                elif transport == "http":
                    await server.run_http(
                        host=host,
                        port=port,
                        cors_enabled=cors,
                    )
                elif transport == "sse":
                    await server.run_sse(
                        host=host,
                        port=port,
                        cors_enabled=cors,
                    )

            except KeyboardInterrupt:
                console.print("\n[yellow]Received interrupt signal, shutting down...[/yellow]")
                await server.shutdown()
                format_info("Server stopped gracefully")
            except SimplyMCPError as e:
                format_error(f"MCP Error: {e}", "Server Error")
                sys.exit(1)
            except Exception as e:
                format_error(f"Unexpected error: {e}", "Fatal Error")
                import traceback
                console.print("[dim]" + traceback.format_exc() + "[/dim]")
                sys.exit(1)

        # Run the async server
        try:
            asyncio.run(run_server())
        except KeyboardInterrupt:
            console.print("\n[dim]Server stopped[/dim]")

    except Exception as e:
        format_error(f"Fatal error: {e}", "Error")
        import traceback
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


__all__ = ["run"]
