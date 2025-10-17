#!/usr/bin/env python3
"""Production-Grade HTTP Server Demo with Polish Layer Features.

This demo showcases all production features of the HTTP transport:
- Configuration from YAML file or environment variables
- Prometheus metrics collection
- Security headers and CORS
- HTTPS/TLS support
- Request validation and protection
- Rate limiting and authentication
- Graceful shutdown
- Structured JSON logging

Usage:
    # Using config file
    python http_server_production.py --config config.yaml

    # Using environment variables
    export MCP_HTTP_SERVER__PORT=9000
    export MCP_HTTP_AUTH__ENABLED=true
    python http_server_production.py

    # Development mode (minimal security)
    python http_server_production.py --dev

Requirements:
    pip install simply-mcp fastapi uvicorn prometheus-client pyyaml
"""

import asyncio
import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for demo imports
sys.path.insert(0, str(Path(__file__).parent))

from simply_mcp import BuildMCPServer
from simply_mcp.core.http_config import HttpConfig
from simply_mcp.transports.http_transport import HttpTransport
from simply_mcp.core.auth import ApiKey
from simply_mcp.core.logger import setup_logger
from simply_mcp.core.config import LogConfigModel


# Define tools for the demo
def echo(message: str, repeat: int = 1) -> dict[str, Any]:
    """Echo a message, optionally repeating it.

    Args:
        message: Message to echo
        repeat: Number of times to repeat (1-10)

    Returns:
        Dict with echoed message and metadata
    """
    if repeat < 1 or repeat > 10:
        raise ValueError("repeat must be between 1 and 10")

    result = "\n".join([message] * repeat)
    return {
        "message": result,
        "original": message,
        "repeated": repeat,
        "length": len(result),
    }


def get_time() -> dict[str, Any]:
    """Get current server time.

    Returns:
        Dict with current time information
    """
    import datetime

    now = datetime.datetime.now()
    return {
        "timestamp": now.isoformat(),
        "unix": now.timestamp(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": str(now.tzinfo),
    }


def calculate(expression: str) -> dict[str, Any]:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., "2 + 2", "10 * 5")

    Returns:
        Dict with calculation result

    Raises:
        ValueError: If expression is invalid or unsafe
    """
    # Only allow safe characters
    allowed = set("0123456789+-*/() .")
    if not all(c in allowed for c in expression):
        raise ValueError("Expression contains invalid characters")

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        }
    except Exception as e:
        raise ValueError(f"Invalid expression: {e}")


def create_development_config() -> HttpConfig:
    """Create development configuration with minimal security.

    Returns:
        HttpConfig for development
    """
    return HttpConfig(
        environment="development",
        server={
            "host": "127.0.0.1",
            "port": 8000,
        },
        tls={
            "enabled": False,
        },
        auth={
            "enabled": False,
        },
        rate_limit={
            "enabled": False,
        },
        monitoring={
            "prometheus_enabled": True,
            "log_requests": True,
            "log_responses": True,
        },
        cors={
            "enabled": True,
            "allow_origins": ["*"],
        },
        security={
            "security_headers": False,
            "request_timeout": 300,  # Long timeout for debugging
        },
        logging={
            "level": "DEBUG",
            "format": "text",
            "enable_console": True,
        },
    )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production HTTP Server with Polish Layer Features"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode (minimal security)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port to bind to (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    if args.dev:
        print("Running in development mode")
        config = create_development_config()
    elif args.config:
        print(f"Loading configuration from {args.config}")
        try:
            config = HttpConfig.from_file(args.config)
        except FileNotFoundError:
            print(f"Error: Configuration file not found: {args.config}")
            print("You can create one from config.example.yaml")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)
    else:
        print("Loading configuration from environment variables")
        print("Set MCP_HTTP_* environment variables to configure")
        try:
            config = HttpConfig.from_env()
        except Exception as e:
            print(f"Error loading configuration from environment: {e}")
            print("Using default configuration")
            config = HttpConfig()

    # Override port if specified
    if args.port:
        config.server.port = args.port

    # Setup logging
    log_config = LogConfigModel(
        level=config.logging.level,
        format=config.logging.format,
        enable_console=config.logging.enable_console,
        file=config.logging.file,
    )
    logger = setup_logger(log_config)

    logger.info(
        "Starting production HTTP server",
        extra={
            "context": {
                "environment": config.environment,
                "host": config.server.host,
                "port": config.server.port,
                "auth": config.auth.enabled,
                "rate_limit": config.rate_limit.enabled,
                "tls": config.tls.enabled,
                "metrics": config.monitoring.prometheus_enabled,
            }
        },
    )

    # Create MCP server
    mcp = BuildMCPServer(
        name="production-demo",
        version="1.0.0",
        description="Production HTTP Server Demo with Polish Layer",
    )

    # Register tools
    mcp.tool()(echo)
    mcp.tool()(get_time)
    mcp.tool()(calculate)

    logger.info(f"Registered {len(mcp.list_tools())} tools")

    # Setup API keys for authentication (if enabled)
    api_keys: Optional[list[ApiKey]] = None
    if config.auth.enabled:
        # Try to load from environment first
        env_key = os.getenv(config.auth.key_env_var)
        if env_key:
            api_keys = [
                ApiKey(
                    key=env_key,
                    name="env-key",
                    description="API key from environment",
                    rate_limit=config.rate_limit.default_limit,
                    window_seconds=config.rate_limit.window_seconds,
                )
            ]
            logger.info("Loaded API key from environment variable")
        else:
            # Demo keys for testing
            api_keys = [
                ApiKey(
                    key="demo-key-12345",
                    name="demo-key",
                    description="Demo API key",
                    rate_limit=100,
                    window_seconds=60,
                ),
                ApiKey(
                    key="test-key-67890",
                    name="test-key",
                    description="Test API key",
                    rate_limit=50,
                    window_seconds=60,
                ),
            ]
            logger.warning(
                "Using demo API keys (not for production!)",
                extra={
                    "context": {
                        "keys": [key.name for key in api_keys]
                    }
                },
            )

    # Create HTTP transport with configuration
    transport = HttpTransport(
        server=mcp,
        config=config,
        api_keys=api_keys,
    )

    # Print startup banner
    protocol = "https" if config.tls.enabled else "http"
    print("\n" + "=" * 70)
    print(f"  Production HTTP Server - {config.environment.upper()}")
    print("=" * 70)
    print(f"  Server:        {mcp.name} v{mcp.version}")
    print(f"  Endpoint:      {protocol}://{config.server.host}:{config.server.port}")
    print(f"  Environment:   {config.environment}")
    print(f"  Auth:          {'Enabled' if config.auth.enabled else 'Disabled'}")
    print(f"  Rate Limiting: {'Enabled' if config.rate_limit.enabled else 'Disabled'}")
    print(f"  TLS/HTTPS:     {'Enabled' if config.tls.enabled else 'Disabled'}")
    print(f"  Metrics:       {protocol}://{config.server.host}:{config.server.port}{config.monitoring.prometheus_path}")
    print(f"  Health:        {protocol}://{config.server.host}:{config.server.port}{config.monitoring.health_path}")
    print(f"  Tools:         {protocol}://{config.server.host}:{config.server.port}/tools")
    print("=" * 70)

    if api_keys and not os.getenv(config.auth.key_env_var):
        print("\n  Demo API Keys (for testing):")
        for key in api_keys:
            print(f"    - {key.name}: {key.key}")
        print("\n  Example request:")
        print(f"    curl -H 'Authorization: Bearer {api_keys[0].key}' \\")
        print(f"         {protocol}://{config.server.host}:{config.server.port}/tools/echo \\")
        print(f"         -d '{{\"message\":\"Hello, World!\"}}'")

    print("=" * 70 + "\n")
    print("Press Ctrl+C to stop the server\n")

    # Start server
    try:
        async with transport:
            # Keep server running
            await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Server shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
