"""Stdio transport helpers for Simply-MCP."""

import asyncio
from typing import Optional

from simply_mcp.core.config import SimplyMCPConfig
from simply_mcp.core.server import SimplyMCPServer


async def run_stdio_server(
    server: SimplyMCPServer,
    config: Optional[SimplyMCPConfig] = None
) -> None:
    """Run MCP server with stdio transport.

    This is a convenience function that initializes and runs a server
    with stdio transport - the most common MCP transport mode.

    Args:
        server: The SimplyMCPServer instance to run
        config: Optional configuration (uses server config if not provided)

    Example:
        >>> server = SimplyMCPServer()
        >>> # Register tools/prompts/resources
        >>> await run_stdio_server(server)
    """
    await server.initialize()
    await server.run_stdio()


__all__ = ["run_stdio_server"]
