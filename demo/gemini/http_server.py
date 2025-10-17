#!/usr/bin/env python3
"""Gemini MCP Server - HTTP Transport Demo

This demonstrates running the Gemini MCP server with HTTP transport.
This is the foundation layer - basic HTTP functionality without auth/rate limiting.

The HTTP server exposes all Gemini tools as REST endpoints:
- POST /tools/upload_file
- POST /tools/generate_content
- POST /tools/start_chat
- POST /tools/send_message
- POST /tools/list_files
- POST /tools/delete_file

Usage:
    # Set API key
    export GEMINI_API_KEY="your-api-key"

    # Install HTTP dependencies
    pip install fastapi uvicorn

    # Run HTTP server
    python demo/gemini/http_server.py

    # Or with custom port
    python demo/gemini/http_server.py --port 8080

Example API calls:
    # Health check
    curl http://localhost:8000/health

    # List all tools
    curl http://localhost:8000/tools

    # Upload a file
    curl -X POST http://localhost:8000/tools/upload_file \\
      -H "Content-Type: application/json" \\
      -d '{"file_uri": "/path/to/file.pdf", "display_name": "My Document"}'

    # Generate content
    curl -X POST http://localhost:8000/tools/generate_content \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Explain quantum computing", "model": "gemini-2.5-flash"}'

    # Start chat session
    curl -X POST http://localhost:8000/tools/start_chat \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "chat-001", "initial_message": "Hello!", "model": "gemini-2.5-flash"}'

    # Send message to chat
    curl -X POST http://localhost:8000/tools/send_message \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "chat-001", "message": "Tell me a joke"}'

    # List files
    curl -X POST http://localhost:8000/tools/list_files \\
      -H "Content-Type: application/json" \\
      -d '{}'

    # Delete file
    curl -X POST http://localhost:8000/tools/delete_file \\
      -H "Content-Type: application/json" \\
      -d '{"file_name": "files/abc123"}'

Features (Foundation Layer):
    - HTTP REST endpoints for all tools
    - JSON request/response handling
    - Basic error handling
    - Health check endpoint
    - Structured logging

NOT included in foundation (coming in feature layer):
    - Authentication
    - Rate limiting
    - API keys
    - Request throttling
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from demo.gemini.server import create_gemini_server

# Check if HTTP transport is available
try:
    from simply_mcp.transports.http_transport import HttpTransport

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False
    HttpTransport = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)

logger = logging.getLogger(__name__)


def check_dependencies() -> tuple[bool, list[str]]:
    """Check if all required dependencies are installed.

    Returns:
        Tuple of (all_ok, missing_packages)
    """
    missing = []

    # Check Gemini SDK
    try:
        import google.genai  # noqa: F401
    except ImportError:
        missing.append("google-genai")

    # Check HTTP transport dependencies
    if not HTTP_AVAILABLE:
        try:
            import fastapi  # noqa: F401
        except ImportError:
            missing.append("fastapi")

        try:
            import uvicorn  # noqa: F401
        except ImportError:
            missing.append("uvicorn")

    return len(missing) == 0, missing


async def main() -> None:
    """Main entry point for HTTP server."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run Gemini MCP Server with HTTP transport"
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
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    args = parser.parse_args()

    # Print banner
    print("=" * 70)
    print("Gemini MCP Server - HTTP Transport (Foundation Layer)")
    print("=" * 70)
    print()

    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print("ERROR: Missing required dependencies:")
        for package in missing:
            print(f"  - {package}")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        sys.exit(1)

    # Check for API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set your API key: export GEMINI_API_KEY='your-api-key'")
        print()
        sys.exit(1)

    # Create Gemini server
    logger.info("Creating Gemini MCP server...")
    mcp = create_gemini_server()

    # Initialize server
    logger.info("Initializing server...")
    await mcp.initialize()

    # Print server info
    print("Server Configuration:")
    print(f"  Name: {mcp.name}")
    print(f"  Version: {mcp.version}")
    print(f"  Transport: HTTP")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print()
    print("Available Tools:")
    tools = mcp.list_tools()
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool}")
    print()
    print("API Endpoints:")
    print(f"  Health:     GET  http://{args.host}:{args.port}/health")
    print(f"  List Tools: GET  http://{args.host}:{args.port}/tools")
    print(f"  Call Tool:  POST http://{args.host}:{args.port}/tools/{{tool_name}}")
    print()
    print("Features (Foundation Layer):")
    print("  - HTTP REST API for all tools")
    print("  - JSON request/response handling")
    print("  - Basic error handling")
    print("  - Structured logging")
    print()
    print("NOT included (coming in feature layer):")
    print("  - Authentication")
    print("  - Rate limiting")
    print("  - API keys")
    print()
    print("=" * 70)
    print()
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    # Create HTTP transport
    transport = HttpTransport(
        server=mcp,
        host=args.host,
        port=args.port,
    )

    # Start server
    try:
        await transport.start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        print()
        print("Shutting down server...")

    finally:
        # Stop transport
        await transport.stop()
        print("Server stopped.")
        print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"\nERROR: {e}")
        sys.exit(1)
