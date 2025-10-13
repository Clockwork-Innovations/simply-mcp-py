#!/usr/bin/env python3
"""HTTP Server Example for Simply-MCP.

This example demonstrates how to create an MCP server that runs over HTTP
transport with JSON-RPC 2.0 protocol. It includes multiple tools, CORS
configuration, and health check endpoints.

Usage:
    # Run with default settings (port 3000)
    python examples/http_server.py

    # Run with custom port
    python examples/http_server.py --port 8080

    # Run with custom host
    python examples/http_server.py --host localhost --port 8080

    # Test the server:
    curl http://localhost:3000/
    curl http://localhost:3000/health
    curl -X POST http://localhost:3000/mcp \\
      -H "Content-Type: application/json" \\
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

    # Call a tool:
    curl -X POST http://localhost:3000/mcp \\
      -H "Content-Type: application/json" \\
      -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":5,"b":3}}}'

Requirements:
    - simply-mcp (with HTTP transport dependencies)
    - aiohttp>=3.9.0

Features Demonstrated:
    - HTTP transport with JSON-RPC 2.0
    - CORS support for web clients
    - Multiple tools with different return types
    - Health check endpoint
    - Graceful shutdown handling
    - Custom port and host configuration
"""

import asyncio
import sys

from simply_mcp import SimplyMCP

# Create MCP server with HTTP transport
mcp = SimplyMCP(
    name="http-demo-server",
    version="1.0.0",
    description="Demo MCP server with HTTP transport",
)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of a / b

    Raises:
        ValueError: If denominator is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
def get_statistics() -> dict[str, int]:
    """Get server statistics.

    Returns:
        Dictionary with server statistics
    """
    return {
        "tools_count": 5,
        "uptime_seconds": 3600,
        "requests_handled": 42,
    }


@mcp.tool()
def list_items() -> list[str]:
    """Get a list of sample items.

    Returns:
        List of sample items
    """
    return ["apple", "banana", "cherry", "date", "elderberry"]


def main() -> None:
    """Run the HTTP server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Simply-MCP HTTP server demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python http_server.py

  # Run on custom port
  python http_server.py --port 8080

  # Run on localhost only
  python http_server.py --host localhost

  # Disable CORS
  python http_server.py --no-cors

  # Custom CORS origins
  python http_server.py --cors-origins http://localhost:3000 http://localhost:8080

Testing:
  # Get server info
  curl http://localhost:3000/

  # Check health
  curl http://localhost:3000/health

  # List tools
  curl -X POST http://localhost:3000/mcp \\
    -H "Content-Type: application/json" \\
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'

  # Call add tool
  curl -X POST http://localhost:3000/mcp \\
    -H "Content-Type: application/json" \\
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"add","arguments":{"a":5,"b":3}}}'
        """,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to bind to (default: 3000)",
    )
    parser.add_argument(
        "--cors/--no-cors",
        default=True,
        help="Enable/disable CORS (default: enabled)",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="*",
        default=None,
        help="Allowed CORS origins (default: all)",
    )

    args = parser.parse_args()

    # Display startup information
    print("=" * 70)
    print(f"Simply-MCP HTTP Server - {mcp.server.config.server.name}")
    print("=" * 70)
    print(f"Version: {mcp.server.config.server.version}")
    print(f"Description: {mcp.server.config.server.description}")
    print("\nHTTP Server Configuration:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  CORS: {'enabled' if args.cors else 'disabled'}")
    if args.cors and args.cors_origins:
        print(f"  CORS Origins: {', '.join(args.cors_origins)}")
    print("\nEndpoints:")
    print(f"  - http://{args.host}:{args.port}/ (server info)")
    print(f"  - http://{args.host}:{args.port}/health (health check)")
    print(f"  - http://{args.host}:{args.port}/mcp (JSON-RPC 2.0)")
    print("\nTools Available:")
    for tool_name in ["add", "multiply", "divide", "get_statistics", "list_items"]:
        print(f"  - {tool_name}")
    print("\n" + "=" * 70)
    print("Server is starting... Press Ctrl+C to stop")
    print("=" * 70 + "\n")

    # Run server
    async def run_server() -> None:
        """Run the HTTP server asynchronously."""
        try:
            await mcp.run_http(
                host=args.host,
                port=args.port,
                cors_enabled=args.cors,
                cors_origins=args.cors_origins,
            )
        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
