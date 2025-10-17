# HTTP Transport Foundation Layer - Setup Guide

This guide covers the foundation layer of HTTP transport for Simply-MCP. The foundation layer provides basic HTTP REST API functionality for exposing MCP tools.

> **Note on Transport Types**: Simply-MCP includes two HTTP transport implementations:
> - `HTTPTransport` (in `http.py`) - Full-featured JSON-RPC 2.0 MCP protocol transport with auth/rate limiting
> - `HttpTransport` (in `http_transport.py`) - Foundation-layer REST API transport (this document)
>
> This guide documents the foundation-layer `HttpTransport` which exposes tools as simple REST endpoints.
> Use this for simpler use cases or as a learning foundation before adding advanced features.

## What's Included in Foundation Layer

The foundation layer provides:

- ✅ HTTP REST endpoints for all MCP tools
- ✅ JSON request/response handling
- ✅ Basic request validation
- ✅ Error handling with proper HTTP status codes
- ✅ Health check endpoint
- ✅ Structured logging
- ✅ Configurable host and port
- ✅ Async/await support

## What's NOT Included (Coming in Feature Layer)

The following features will be added in the feature layer:

- ❌ Authentication/authorization
- ❌ Rate limiting
- ❌ API keys
- ❌ Request throttling
- ❌ Advanced middleware
- ❌ Request caching

## Installation

### 1. Install Required Dependencies

The HTTP transport requires FastAPI and Uvicorn:

```bash
pip install fastapi uvicorn httpx
```

Or install simply-mcp with HTTP support:

```bash
pip install "simply-mcp[http]"  # If added to optional dependencies
```

### 2. Verify Installation

Check that dependencies are installed:

```bash
python -c "import fastapi, uvicorn; print('HTTP transport dependencies OK')"
```

## Basic Usage

### Creating an HTTP Server

```python
import asyncio
from simply_mcp import BuildMCPServer
from simply_mcp.transports.http_transport import HttpTransport

# Create MCP server
mcp = BuildMCPServer(name="my-server", version="1.0.0")

# Register tools
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@mcp.tool()
async def async_multiply(a: int, b: int) -> int:
    """Multiply two numbers asynchronously."""
    await asyncio.sleep(0.1)  # Simulate async work
    return a * b

# Initialize server
async def main():
    await mcp.initialize()

    # Create HTTP transport
    transport = HttpTransport(
        server=mcp,
        host="0.0.0.0",
        port=8000
    )

    # Start server
    await transport.start()

    print("Server running on http://0.0.0.0:8000")
    print("Press Ctrl+C to stop")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await transport.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Using as Context Manager

The HTTP transport can be used as an async context manager:

```python
async def main():
    await mcp.initialize()

    async with HttpTransport(server=mcp, host="0.0.0.0", port=8000) as transport:
        print("Server is running")
        # Server automatically starts and stops
        await asyncio.sleep(60)  # Run for 60 seconds
```

## Configuration Options

### Host and Port

```python
# Bind to localhost only
transport = HttpTransport(server=mcp, host="127.0.0.1", port=8000)

# Bind to all interfaces
transport = HttpTransport(server=mcp, host="0.0.0.0", port=8000)

# Custom port
transport = HttpTransport(server=mcp, host="0.0.0.0", port=9999)
```

## API Endpoints

Once the server is running, the following endpoints are available:

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "server": "my-server",
  "version": "1.0.0"
}
```

### List Tools

```bash
GET /tools
```

Response:
```json
{
  "tools": ["add", "multiply", "..."],
  "count": 3
}
```

### Call a Tool

```bash
POST /tools/{tool_name}
Content-Type: application/json

{
  "param1": "value1",
  "param2": "value2"
}
```

Response:
```json
{
  "success": true,
  "tool": "tool_name",
  "result": {...}
}
```

## Example: Gemini HTTP Server

The project includes a complete example using the Gemini MCP server:

```bash
# Set API key
export GEMINI_API_KEY="your-api-key"

# Run HTTP server
python demo/gemini/http_server.py

# Or with custom port
python demo/gemini/http_server.py --port 8080
```

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# List all tools
curl http://localhost:8000/tools

# Generate content
curl -X POST http://localhost:8000/tools/generate_content \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "model": "gemini-2.5-flash"
  }'

# Upload a file
curl -X POST http://localhost:8000/tools/upload_file \
  -H "Content-Type: application/json" \
  -d '{
    "file_uri": "/path/to/document.pdf",
    "display_name": "My Document"
  }'

# Start chat session
curl -X POST http://localhost:8000/tools/start_chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat-001",
    "initial_message": "Hello! Can you help me?",
    "model": "gemini-2.5-flash"
  }'

# Send message to chat
curl -X POST http://localhost:8000/tools/send_message \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat-001",
    "message": "Tell me about machine learning"
  }'
```

## Error Handling

The HTTP transport provides proper HTTP status codes:

- `200 OK` - Successful tool execution
- `400 Bad Request` - Invalid parameters or malformed JSON
- `404 Not Found` - Tool does not exist
- `500 Internal Server Error` - Tool execution failed

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Logging

The HTTP transport uses structured logging. All requests and responses are logged at appropriate levels:

- `INFO` - Server startup, tool calls, successful operations
- `WARNING` - Invalid requests, missing parameters
- `ERROR` - Tool execution failures, server errors
- `DEBUG` - Detailed request/response data

## Testing

Run the HTTP transport tests:

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run HTTP transport tests
pytest tests/test_http_transport_foundation.py -v

# Run with coverage
pytest tests/test_http_transport_foundation.py --cov=simply_mcp.transports.http_transport
```

## Known Limitations (Foundation Layer)

These limitations will be addressed in the feature layer:

1. **No Authentication** - All endpoints are publicly accessible
2. **No Rate Limiting** - No protection against abuse or overload
3. **No Request Validation** - Limited parameter validation beyond basic type checking
4. **No CORS Configuration** - Cross-origin requests not configured
5. **No SSL/TLS** - HTTP only (not HTTPS)
6. **No Request Logging** - Limited request/response logging
7. **No Metrics** - No performance metrics or monitoring

## Next Steps

After validating the foundation layer:

1. **Feature Layer** - Add authentication, rate limiting, and advanced features
2. **Polish Layer** - Add production-ready features, optimization, and documentation
3. **Production Deployment** - Deploy with proper security and monitoring

## Troubleshooting

### Port Already in Use

```bash
# Error: Address already in use
# Solution: Use a different port or kill the process using the port

# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use a different port
python demo/gemini/http_server.py --port 8001
```

### Import Errors

```bash
# Error: No module named 'fastapi'
# Solution: Install HTTP dependencies

pip install fastapi uvicorn httpx
```

### Connection Refused

If you can't connect to the server:

1. Check server is running: `curl http://localhost:8000/health`
2. Check correct port is being used
3. Check firewall settings
4. Try binding to `127.0.0.1` instead of `0.0.0.0`

## Support

For issues or questions:

1. Check the test suite: `tests/test_http_transport_foundation.py`
2. Review example server: `demo/gemini/http_server.py`
3. Check server logs for error messages
4. Open an issue on GitHub

## Summary

The HTTP Transport Foundation Layer provides basic HTTP REST API functionality for Simply-MCP servers. It's designed to be simple, reliable, and easy to use as a foundation for more advanced features.

**Key Points:**
- ✅ Basic HTTP functionality is complete and tested
- ✅ All MCP tools are exposed as REST endpoints
- ✅ Proper error handling and logging
- ❌ Authentication and rate limiting come in feature layer
- ❌ Production features come in polish layer

This foundation layer is ready for testing and validation before proceeding to the feature layer.
