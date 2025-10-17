# HTTP Authentication and Rate Limiting

This document describes the authentication and rate limiting features for the HTTP transport in Simply-MCP.

## Overview

The HTTP transport supports optional authentication and rate limiting to secure your MCP tools when exposed via HTTP endpoints. These are **feature layer** additions that work transparently with the existing foundation layer.

### Why Authentication?

- **Security**: Prevent unauthorized access to your MCP tools
- **Access Control**: Control who can use which tools
- **Audit Trail**: Track API usage per client

### Why Rate Limiting?

- **Protection**: Prevent abuse and DoS attacks
- **Fair Usage**: Ensure equitable access across clients
- **Cost Control**: Limit usage to stay within API quotas

## Key Features

- ✅ Bearer token authentication (RFC 6750 compliant)
- ✅ API key management with per-key configuration
- ✅ Token bucket rate limiting algorithm
- ✅ Per-key rate limits
- ✅ Rate limit headers in responses
- ✅ Backward compatible (both features are optional)
- ✅ Clean separation from foundation layer
- ✅ Zero breaking changes to existing code

## Quick Start

### Basic Setup (No Auth)

The HTTP transport works without authentication for backward compatibility:

```python
from simply_mcp import BuildMCPServer
from simply_mcp.transports.http_transport import HttpTransport

# Create server
mcp = BuildMCPServer(name="my-server", version="1.0.0")

# Register tools...
@mcp.tool()
def hello(name: str) -> str:
    return f"Hello, {name}!"

# Create HTTP transport (no auth)
transport = HttpTransport(server=mcp, host="0.0.0.0", port=8000)
await transport.start()
```

### With Authentication

Enable authentication by providing API keys:

```python
from simply_mcp.core.auth import ApiKey
from simply_mcp.transports.http_transport import HttpTransport

# Define API keys
api_keys = [
    ApiKey(
        key="sk_prod_abc123",
        name="Production Key",
        rate_limit=100,
        window_seconds=3600,
    ),
    ApiKey(
        key="sk_test_xyz789",
        name="Test Key",
        rate_limit=10,
        window_seconds=60,
    ),
]

# Create HTTP transport with auth
transport = HttpTransport(
    server=mcp,
    enable_auth=True,
    api_keys=api_keys,
)
await transport.start()
```

### With Authentication + Rate Limiting

Enable both features for full protection:

```python
transport = HttpTransport(
    server=mcp,
    enable_auth=True,
    enable_rate_limiting=True,
    api_keys=api_keys,
)
await transport.start()
```

## Configuration

### API Key Configuration

Each API key has the following properties:

```python
ApiKey(
    key="sk_prod_abc123",          # The actual API key (required)
    name="Production Key",          # Human-readable name (required)
    rate_limit=100,                 # Max requests per window (default: 100)
    window_seconds=3600,            # Time window in seconds (default: 3600)
    enabled=True,                   # Whether key is active (default: True)
)
```

### Loading Keys from Environment

You can load API keys from environment variables in JSON format:

```bash
export MCP_API_KEYS='{
  "keys": [
    {
      "key": "sk_prod_abc123",
      "name": "Production Key",
      "rate_limit": 100,
      "window_seconds": 3600
    },
    {
      "key": "sk_test_xyz789",
      "name": "Test Key",
      "rate_limit": 10,
      "window_seconds": 60
    }
  ]
}'
```

Then load in your code:

```python
from simply_mcp.core.auth import ApiKeyManager

manager = ApiKeyManager()
manager.load_from_env("MCP_API_KEYS")
```

### Rate Limit Configuration

Rate limits are configured per API key using the token bucket algorithm:

- **max_requests**: Maximum number of requests allowed
- **window_seconds**: Time window for the limit (in seconds)

Example configurations:

```python
# 100 requests per hour
ApiKey(key="key1", name="Standard", rate_limit=100, window_seconds=3600)

# 10 requests per minute
ApiKey(key="key2", name="Limited", rate_limit=10, window_seconds=60)

# 1000 requests per day
ApiKey(key="key3", name="Premium", rate_limit=1000, window_seconds=86400)
```

## Making Authenticated Requests

### Using curl

Include the API key in the `Authorization` header with `Bearer` scheme:

```bash
# List tools
curl -H "Authorization: Bearer sk_prod_abc123" \
     http://localhost:8000/tools

# Execute a tool
curl -X POST http://localhost:8000/tools/hello \
     -H "Authorization: Bearer sk_prod_abc123" \
     -H "Content-Type: application/json" \
     -d '{"name": "World"}'

# Check rate limit headers
curl -v -H "Authorization: Bearer sk_prod_abc123" \
     http://localhost:8000/tools
```

### Using Python

```python
import httpx

headers = {
    "Authorization": "Bearer sk_prod_abc123",
    "Content-Type": "application/json",
}

# List tools
response = httpx.get("http://localhost:8000/tools", headers=headers)
print(response.json())

# Execute tool
response = httpx.post(
    "http://localhost:8000/tools/hello",
    headers=headers,
    json={"name": "World"}
)
print(response.json())
```

### Using JavaScript

```javascript
const headers = {
    'Authorization': 'Bearer sk_prod_abc123',
    'Content-Type': 'application/json'
};

// List tools
const response = await fetch('http://localhost:8000/tools', { headers });
const tools = await response.json();

// Execute tool
const result = await fetch('http://localhost:8000/tools/hello', {
    method: 'POST',
    headers,
    body: JSON.stringify({ name: 'World' })
});
```

## Rate Limit Headers

When rate limiting is enabled, all responses include these headers:

- **X-RateLimit-Limit**: Maximum requests allowed in window
- **X-RateLimit-Remaining**: Requests remaining in current window
- **X-RateLimit-Reset**: Unix timestamp when limit resets

Example response:

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1697558400
Content-Type: application/json

{
  "tools": ["hello", "goodbye"],
  "count": 2
}
```

When rate limit is exceeded:

```
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1697558400
Retry-After: 45
Content-Type: application/json

{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Retry after 44.5 seconds",
  "retry_after": 44.5
}
```

## Error Responses

### 401 Unauthorized - Missing Authorization

```json
{
  "error": "Unauthorized",
  "detail": "Missing or invalid Authorization header. Expected format: 'Bearer <token>'"
}
```

### 401 Unauthorized - Invalid Token

```json
{
  "error": "Unauthorized",
  "detail": "Invalid API key"
}
```

### 429 Too Many Requests

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Retry after 44.5 seconds",
  "retry_after": 44.5
}
```

## Rate Limiting Behavior

### Token Bucket Algorithm

The rate limiting system uses the **token bucket algorithm**:

1. Each API key has a "bucket" that holds tokens
2. The bucket starts full with `max_requests` tokens
3. Each request consumes 1 token
4. Tokens refill at a constant rate: `max_requests / window_seconds`
5. Requests are allowed only if tokens are available

**Example**: For a limit of 100 requests per hour:
- Bucket starts with 100 tokens
- Tokens refill at ~0.0278 tokens/second
- After consuming all 100 tokens, it takes 1 hour to fully refill
- But you can make requests as tokens become available (smooth rate limiting)

### Benefits of Token Bucket

- **Smooth rate limiting**: No sudden blocking at window boundaries
- **Burst tolerance**: Can handle bursts up to bucket capacity
- **Gradual refill**: Tokens become available gradually over time
- **Fair distribution**: Consistent behavior across time

### Rate Limit Reset

The `X-RateLimit-Reset` header shows when the bucket will be **full again**, not when you can make the next request. You can make requests as soon as tokens are available.

## Special Endpoints

### Health Check

The `/health` endpoint **always bypasses authentication** to allow monitoring without credentials:

```bash
curl http://localhost:8000/health
# No Authorization header needed
```

Response:

```json
{
  "status": "healthy",
  "server": "my-server",
  "version": "1.0.0"
}
```

## Advanced Usage

### Dynamic API Key Management

You can add/remove API keys at runtime:

```python
from simply_mcp.core.auth import ApiKey

# Add a new key
new_key = ApiKey(
    key="sk_new_key",
    name="New Key",
    rate_limit=50,
    window_seconds=3600,
)
transport.api_key_manager.add_key(new_key)

# Add rate limit for the new key
from simply_mcp.core.rate_limit import RateLimitConfig

rate_config = RateLimitConfig(
    max_requests=50,
    window_seconds=3600,
)
transport.rate_limiter.add_key("sk_new_key", rate_config)

# Remove a key
transport.api_key_manager.remove_key("sk_old_key")
transport.rate_limiter.remove_key("sk_old_key")
```

### Checking Rate Limit Status

You can check rate limit status without consuming tokens:

```python
# Get status for a key
status = transport.rate_limiter.get_status("sk_prod_abc123")
if status:
    print(f"Remaining: {status.remaining}/{status.limit}")
    print(f"Resets at: {status.reset_at}")
```

### Resetting Rate Limits

For testing or administrative purposes, you can reset rate limits:

```python
# Reset rate limit for a specific key
transport.rate_limiter.reset_key("sk_test_xyz789")
```

### Disabling Keys

You can temporarily disable a key without removing it:

```python
# Disable a key
key_info = transport.api_key_manager.get_key_info("sk_prod_abc123")
if key_info:
    key_info.enabled = False

# Re-enable later
key_info.enabled = True
```

## Complete Example

Here's a complete example with all features:

```python
#!/usr/bin/env python3
import asyncio
import os
from simply_mcp import BuildMCPServer
from simply_mcp.core.auth import ApiKey
from simply_mcp.transports.http_transport import HttpTransport

async def main():
    # Create MCP server
    mcp = BuildMCPServer(
        name="secure-server",
        version="1.0.0",
        description="MCP server with authentication and rate limiting"
    )

    # Register tools
    @mcp.tool()
    def hello(name: str) -> str:
        """Say hello to someone."""
        return f"Hello, {name}!"

    @mcp.tool()
    def goodbye(name: str) -> str:
        """Say goodbye to someone."""
        return f"Goodbye, {name}!"

    # Configure API keys
    api_keys = [
        ApiKey(
            key=os.getenv("PROD_API_KEY", "sk_prod_default"),
            name="Production",
            rate_limit=1000,
            window_seconds=3600,  # 1000 requests per hour
        ),
        ApiKey(
            key=os.getenv("TEST_API_KEY", "sk_test_default"),
            name="Testing",
            rate_limit=100,
            window_seconds=3600,  # 100 requests per hour
        ),
    ]

    # Create HTTP transport with auth and rate limiting
    transport = HttpTransport(
        server=mcp,
        host="0.0.0.0",
        port=8000,
        enable_auth=True,
        enable_rate_limiting=True,
        api_keys=api_keys,
    )

    # Start server
    print("Starting secure HTTP server...")
    print(f"Available API keys:")
    for key in api_keys:
        print(f"  - {key.name}: {key.rate_limit} req/{key.window_seconds}s")
    print("")
    print("Test with:")
    print(f"  curl -H 'Authorization: Bearer {api_keys[0].key}' http://localhost:8000/tools")
    print("")

    await transport.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        await transport.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

Save as `secure_server.py` and run:

```bash
python secure_server.py
```

Test with curl:

```bash
# List tools (should succeed)
curl -H "Authorization: Bearer sk_prod_default" http://localhost:8000/tools

# Execute tool (should succeed)
curl -X POST http://localhost:8000/tools/hello \
     -H "Authorization: Bearer sk_prod_default" \
     -H "Content-Type: application/json" \
     -d '{"name": "World"}'

# Without auth (should fail with 401)
curl http://localhost:8000/tools

# With wrong key (should fail with 401)
curl -H "Authorization: Bearer wrong_key" http://localhost:8000/tools

# Test rate limiting (make many requests)
for i in {1..105}; do
    curl -H "Authorization: Bearer sk_test_default" http://localhost:8000/health
done
```

## Troubleshooting

### Issue: 401 Unauthorized

**Cause**: Missing or invalid Authorization header

**Solution**:
- Ensure you're including the `Authorization: Bearer <token>` header
- Verify the token is correct
- Check that the key hasn't been disabled

### Issue: 429 Too Many Requests

**Cause**: Rate limit exceeded

**Solution**:
- Wait for the time specified in `Retry-After` header
- Check `X-RateLimit-Reset` for when limit fully resets
- Consider requesting a higher rate limit

### Issue: Tools work without auth when auth is enabled

**Cause**: Middleware not properly configured

**Solution**:
- Verify `enable_auth=True` is set in `HttpTransport` initialization
- Check that API keys were provided
- Ensure you're not hitting the `/health` endpoint (which bypasses auth)

### Issue: Rate limiting not working

**Cause**: Rate limiting not enabled or not configured

**Solution**:
- Verify `enable_rate_limiting=True` is set
- Ensure both `enable_auth=True` and API keys are provided (rate limiting requires auth)
- Check that API keys have rate limit configuration

### Issue: Foundation layer tests failing after upgrade

**Cause**: Breaking changes in the feature layer

**Solution**:
- Feature layer is backward compatible - foundation tests should still pass
- Verify you're not enabling auth/rate limiting in foundation tests
- Report as bug if foundation layer is affected

## Migration Guide

### From Foundation Layer (No Auth)

If you're currently using HTTP transport without authentication:

**Before**:
```python
transport = HttpTransport(server=mcp, host="0.0.0.0", port=8000)
```

**After** (with auth):
```python
api_keys = [ApiKey(key="sk_prod_123", name="Production")]
transport = HttpTransport(
    server=mcp,
    host="0.0.0.0",
    port=8000,
    enable_auth=True,
    api_keys=api_keys,
)
```

**After** (with auth + rate limiting):
```python
api_keys = [
    ApiKey(
        key="sk_prod_123",
        name="Production",
        rate_limit=100,
        window_seconds=3600,
    )
]
transport = HttpTransport(
    server=mcp,
    host="0.0.0.0",
    port=8000,
    enable_auth=True,
    enable_rate_limiting=True,
    api_keys=api_keys,
)
```

### Updating Client Code

**Before** (no auth):
```bash
curl http://localhost:8000/tools/hello -d '{"name": "World"}'
```

**After** (with auth):
```bash
curl -H "Authorization: Bearer sk_prod_123" \
     http://localhost:8000/tools/hello \
     -d '{"name": "World"}'
```

## Best Practices

### API Key Security

1. **Never commit keys to version control**
   - Use environment variables or secure key management
   - Add `.env` files to `.gitignore`

2. **Use descriptive key names**
   - Helps with debugging and auditing
   - Example: "Production-WebApp", "Testing-MobileApp"

3. **Rotate keys regularly**
   - Generate new keys periodically
   - Maintain grace period for transition

4. **Use different keys per environment**
   - Separate keys for production, staging, development
   - Makes it easier to revoke compromised keys

### Rate Limiting

1. **Set appropriate limits**
   - Consider your service capacity
   - Balance between usability and protection
   - Monitor actual usage patterns

2. **Provide clear error messages**
   - Include `retry_after` in responses
   - Document rate limits in API documentation

3. **Use tiered limits**
   - Different limits for different user tiers
   - Example: Free (10/min), Pro (100/min), Enterprise (1000/min)

4. **Monitor rate limit hits**
   - Log rate limit exceeded events
   - Alert on unusual patterns
   - Adjust limits based on data

### Production Deployment

1. **Use HTTPS in production**
   - Bearer tokens should only be sent over HTTPS
   - Use reverse proxy (nginx, Caddy) for TLS termination

2. **Implement key rotation**
   - Support multiple active keys during rotation
   - Provide API for key management

3. **Monitor and alert**
   - Track authentication failures
   - Alert on rate limit abuse
   - Monitor for potential attacks

4. **Document your API**
   - Provide clear authentication instructions
   - Document rate limits for each tier
   - Include example requests

## Foundation Layer Compatibility

The authentication and rate limiting features are **completely optional** and maintain full backward compatibility with the foundation layer:

- ✅ Foundation layer tests still pass (21/22 passing)
- ✅ Existing code works without changes
- ✅ No breaking changes to API
- ✅ Auth and rate limiting are opt-in
- ✅ Health endpoint always works
- ✅ Clean separation of concerns

You can continue using the HTTP transport exactly as before, and add authentication/rate limiting when needed.

## What's Next?

This is the **Feature Layer** implementation. Future enhancements in the **Polish Layer** may include:

- Persistent rate limit storage (Redis, etc.)
- OAuth2 support
- JWT token validation
- API key scoping (per-tool permissions)
- Advanced rate limiting strategies
- WebSocket support with auth
- Metrics and analytics dashboard

## Support

For issues, questions, or feature requests:
- GitHub Issues: [simply-mcp-py/issues](https://github.com/your-org/simply-mcp-py/issues)
- Documentation: [simply-mcp.dev](https://simply-mcp.dev)
- Examples: See `demo/gemini/http_server_with_auth.py`
