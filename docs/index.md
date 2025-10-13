# Simply-MCP-PY

A modern, Pythonic framework for building Model Context Protocol (MCP) servers with multiple API styles.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

## Overview

Simply-MCP-PY is the Python implementation of [simply-mcp-ts](https://github.com/Clockwork-Innovations/simply-mcp-ts), bringing the same ease-of-use and flexibility to the Python ecosystem for building MCP servers. It provides a high-level abstraction over the [Anthropic MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) with a focus on developer experience and productivity.

## Key Features

### Multiple API Styles
Choose the style that fits your workflow:

- **Decorator API** - Clean, declarative class-based approach
- **Functional API** - Programmatic server building with method chaining
- **Interface API** - Pure type-annotated interfaces (coming soon)
- **Builder API** - AI-powered tool development (future)

### Multiple Transports
Run your server anywhere:

- **Stdio** - Standard input/output (default)
- **HTTP** - RESTful HTTP server with session support
- **SSE** - Server-Sent Events for real-time streaming

### Zero Configuration
Get started instantly:

- Auto-detect API style
- Automatic schema generation from type hints
- Sensible defaults for everything
- Optional configuration for advanced use cases

### Developer Experience

- Hot reload with watch mode
- Bundle to standalone executable
- Type-safe with full mypy support
- Comprehensive documentation

### Production Ready

- Rate limiting and authentication
- Progress reporting for long operations
- Binary content support
- Security best practices
- Structured JSON logging

## Quick Start

### Installation

```bash
pip install simply-mcp
```

### Your First Server

```python
# server.py
from simply_mcp import mcp_server, tool

@mcp_server(name="my-server", version="1.0.0")
class MyServer:
    @tool(description="Add two numbers")
    def add(self, a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @tool(description="Greet a user")
    def greet(self, name: str, formal: bool = False) -> str:
        """Generate a greeting."""
        if formal:
            return f"Good day, {name}."
        return f"Hey {name}!"
```

### Run Your Server

```bash
# Run with stdio (default)
simply-mcp run server.py

# Run with HTTP on port 3000
simply-mcp run server.py --http --port 3000

# Run with auto-reload
simply-mcp run server.py --watch
```

## Next Steps

- [Installation Guide](getting-started/installation.md) - Detailed installation instructions
- [Quick Start Tutorial](getting-started/quickstart.md) - 5-minute walkthrough
- [First Server](getting-started/first-server.md) - Build your first MCP server
- [API Reference](api/core/server.md) - Complete API documentation
- [Examples](examples/index.md) - Working code examples

## Requirements

- Python 3.10 or higher
- pip or poetry for package management

## Development Status

Currently in **Alpha** - Core features are implemented and functional, but the API may change. See our [roadmap](https://github.com/Clockwork-Innovations/simply-mcp-py/blob/main/docs/ROADMAP.md) for development progress.

## Support

- [Documentation](https://simply-mcp-py.readthedocs.io)
- [Issue Tracker](https://github.com/Clockwork-Innovations/simply-mcp-py/issues)
- [Discussions](https://github.com/Clockwork-Innovations/simply-mcp-py/discussions)
- [GitHub Repository](https://github.com/Clockwork-Innovations/simply-mcp-py)

## License

MIT License - see [LICENSE](https://github.com/Clockwork-Innovations/simply-mcp-py/blob/main/LICENSE) file for details.

---

Made with by Clockwork Innovations
