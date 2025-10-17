# HTTP Transport Foundation Layer - Quick Start

## Installation

```bash
pip install fastapi uvicorn httpx
```

## Basic Usage

```python
import asyncio
from simply_mcp import BuildMCPServer
from simply_mcp.transports.http_transport import HttpTransport

# Create server
mcp = BuildMCPServer(name="my-server", version="1.0.0")

# Register tools
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Run HTTP server
async def main():
    await mcp.initialize()

    transport = HttpTransport(server=mcp, host="0.0.0.0", port=8000)
    await transport.start()

    print("Server running on http://0.0.0.0:8000")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await transport.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing the Server

```bash
# Health check
curl http://localhost:8000/health

# List tools
curl http://localhost:8000/tools

# Call a tool
curl -X POST http://localhost:8000/tools/add \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

## Running the Gemini Demo

```bash
export GEMINI_API_KEY="your-api-key"
python demo/gemini/http_server.py --port 8000
```

## Running Tests

```bash
pytest tests/test_http_transport_foundation.py -v
```

## Key Files

- **Core Transport**: `src/simply_mcp/transports/http_transport.py`
- **Demo Server**: `demo/gemini/http_server.py`
- **Tests**: `tests/test_http_transport_foundation.py`
- **Docs**: `docs/HTTP_TRANSPORT_SETUP.md`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/tools` | List all tools |
| POST | `/tools/{name}` | Call a tool |
| POST | `/api/{name}` | Call a tool (alternative) |

## What's Included

✅ HTTP REST endpoints
✅ JSON request/response
✅ Error handling
✅ Logging
✅ Async support

## What's NOT Included (Feature Layer)

❌ Authentication
❌ Rate limiting
❌ API keys
❌ SSL/TLS

## Need Help?

- Read full docs: `docs/HTTP_TRANSPORT_SETUP.md`
- Check tests: `tests/test_http_transport_foundation.py`
- Review summary: `HTTP_TRANSPORT_IMPLEMENTATION_SUMMARY.md`
