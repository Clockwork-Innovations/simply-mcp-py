#!/usr/bin/env python3
"""Gemini MCP HTTP Server with Authentication and Rate Limiting.

This demo shows how to use the HTTP transport with authentication
and rate limiting enabled.

Features:
- Bearer token authentication
- Per-key rate limiting
- All Gemini tools accessible via HTTP with auth
- Rate limit headers in responses

Setup:
    1. Set your Gemini API key:
       export GEMINI_API_KEY="your-gemini-api-key"

    2. Set API keys for authentication (JSON format):
       export MCP_API_KEYS='{"keys": [{"key": "sk_test_12345", "name": "Test Key", "rate_limit": 10, "window_seconds": 60}]}'

    3. Run the server:
       python demo/gemini/http_server_with_auth.py

    4. Test with curl:
       # List tools (requires auth)
       curl -H "Authorization: Bearer sk_test_12345" http://localhost:8000/tools

       # Upload a file
       curl -X POST http://localhost:8000/tools/upload_file \\
         -H "Authorization: Bearer sk_test_12345" \\
         -H "Content-Type: application/json" \\
         -d '{"file_uri": "/path/to/file.txt", "display_name": "Test File"}'

       # Generate content
       curl -X POST http://localhost:8000/tools/generate_content \\
         -H "Authorization: Bearer sk_test_12345" \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Hello, how are you?"}'

Rate Limiting:
    - Each API key has its own rate limit
    - Default: 10 requests per minute (configurable)
    - Rate limit info included in response headers:
      - X-RateLimit-Limit: Maximum requests allowed
      - X-RateLimit-Remaining: Requests remaining
      - X-RateLimit-Reset: Unix timestamp when limit resets

Example curl commands:
    # Check rate limit headers
    curl -v -H "Authorization: Bearer sk_test_12345" http://localhost:8000/tools

    # Exceed rate limit (make 11 requests quickly)
    for i in {1..11}; do
        curl -H "Authorization: Bearer sk_test_12345" http://localhost:8000/health
    done
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from simply_mcp import BuildMCPServer
from simply_mcp.core.auth import ApiKey
from simply_mcp.transports.http_transport import HttpTransport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)

logger = logging.getLogger(__name__)


def load_api_keys_from_env() -> list[ApiKey]:
    """Load API keys from environment variable.

    Expects MCP_API_KEYS environment variable with JSON format:
    {
        "keys": [
            {
                "key": "sk_test_12345",
                "name": "Test Key",
                "rate_limit": 10,
                "window_seconds": 60
            }
        ]
    }

    Returns:
        List of ApiKey objects
    """
    json_data = os.getenv("MCP_API_KEYS")

    if not json_data:
        logger.warning("MCP_API_KEYS environment variable not set")
        # Return default test key for demo purposes
        logger.info("Using default test API key: sk_test_12345")
        return [
            ApiKey(
                key="sk_test_12345",
                name="Default Test Key",
                rate_limit=10,
                window_seconds=60,
            )
        ]

    try:
        config = json.loads(json_data)
        keys = []

        for key_config in config.get("keys", []):
            api_key = ApiKey(
                key=key_config["key"],
                name=key_config.get("name", "Unnamed Key"),
                rate_limit=key_config.get("rate_limit", 100),
                window_seconds=key_config.get("window_seconds", 3600),
                enabled=key_config.get("enabled", True),
            )
            keys.append(api_key)

        logger.info(f"Loaded {len(keys)} API keys from environment")
        return keys

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse MCP_API_KEYS: {e}")
        logger.info("Using default test API key: sk_test_12345")
        return [
            ApiKey(
                key="sk_test_12345",
                name="Default Test Key",
                rate_limit=10,
                window_seconds=60,
            )
        ]


async def main() -> None:
    """Run HTTP server with authentication and rate limiting."""
    logger.info("=" * 70)
    logger.info("Gemini MCP HTTP Server with Authentication")
    logger.info("=" * 70)

    # Check for Gemini API key
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("\nERROR: GEMINI_API_KEY environment variable not set")
        print("Please set your API key: export GEMINI_API_KEY='your-api-key'\n")
        return

    # Load API keys for authentication
    api_keys = load_api_keys_from_env()

    # Import and create Gemini server
    try:
        from server import create_gemini_server
    except ImportError:
        # Try alternative import path
        sys.path.insert(0, str(Path(__file__).parent))
        from server import create_gemini_server

    mcp = create_gemini_server()

    # Create HTTP transport with authentication and rate limiting
    transport = HttpTransport(
        server=mcp,
        host="0.0.0.0",
        port=8000,
        enable_auth=True,
        api_keys=api_keys,
        enable_rate_limiting=True,
    )

    logger.info("")
    logger.info("Server Configuration:")
    logger.info(f"  - Host: 0.0.0.0:8000")
    logger.info(f"  - Authentication: ENABLED")
    logger.info(f"  - Rate Limiting: ENABLED")
    logger.info(f"  - API Keys Loaded: {len(api_keys)}")
    logger.info("")

    # Display API keys
    logger.info("Configured API Keys:")
    for api_key in api_keys:
        logger.info(
            f"  - {api_key.name}: "
            f"{api_key.rate_limit} req/{api_key.window_seconds}s"
        )

    logger.info("")
    logger.info("Example curl commands:")
    logger.info("")
    logger.info("  # Health check (no auth required)")
    logger.info("  curl http://localhost:8000/health")
    logger.info("")
    logger.info("  # List tools (auth required)")
    logger.info(f'  curl -H "Authorization: Bearer {api_keys[0].key}" http://localhost:8000/tools')
    logger.info("")
    logger.info("  # Generate content")
    logger.info("  curl -X POST http://localhost:8000/tools/generate_content \\")
    logger.info(f'    -H "Authorization: Bearer {api_keys[0].key}" \\')
    logger.info('    -H "Content-Type: application/json" \\')
    logger.info('    -d \'{"prompt": "Hello, how are you?"}\'')
    logger.info("")
    logger.info("  # Test rate limiting (make multiple requests quickly)")
    logger.info("  for i in {1..15}; do")
    logger.info(f'    curl -H "Authorization: Bearer {api_keys[0].key}" http://localhost:8000/health')
    logger.info("  done")
    logger.info("")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Server starting...")
    logger.info("")

    try:
        # Start server
        await transport.start()

        logger.info("Server is running! Press Ctrl+C to stop.")
        logger.info("")

        # Keep server running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Shutting down server...")
        await transport.stop()
        logger.info("Server stopped.")


if __name__ == "__main__":
    asyncio.run(main())
