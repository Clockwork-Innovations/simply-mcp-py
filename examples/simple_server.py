#!/usr/bin/env python3
"""Simple MCP server example using Simply-MCP.

This example demonstrates:
- Creating a SimplyMCPServer
- Registering tools with handlers
- Running with stdio transport
"""

import asyncio
from simply_mcp.core.server import SimplyMCPServer
from simply_mcp.core.config import get_default_config
from simply_mcp.transports.stdio import run_stdio_server


# Define tool handlers
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def greet(name: str, formal: bool = False) -> str:
    """Generate a greeting message."""
    if formal:
        return f"Good day, {name}."
    return f"Hey {name}!"


async def main():
    """Main entry point."""
    # Create server with default config
    config = get_default_config()
    server = SimplyMCPServer(config)

    # Register add tool
    server.register_tool({
        "name": "add",
        "description": "Add two numbers together",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        },
        "handler": add_numbers
    })

    # Register greet tool
    server.register_tool({
        "name": "greet",
        "description": "Generate a greeting message",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Person's name"},
                "formal": {"type": "boolean", "description": "Use formal greeting", "default": False}
            },
            "required": ["name"]
        },
        "handler": greet
    })

    # Run server with stdio transport
    await run_stdio_server(server)


if __name__ == "__main__":
    asyncio.run(main())
