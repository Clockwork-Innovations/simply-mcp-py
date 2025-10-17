# HTTP Transport Production Deployment Guide

Complete guide for deploying Simply-MCP HTTP transport in production environments with all polish layer features enabled.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Security](#security)
- [Monitoring](#monitoring)
- [TLS/HTTPS Setup](#tlshttps-setup)
- [Performance Tuning](#performance-tuning)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

## Overview

The HTTP transport polish layer adds production-grade features on top of the foundation and feature layers:

**Foundation Layer** (Basic HTTP)
- REST endpoints for MCP tools
- JSON request/response
- Basic error handling
- Structured logging

**Feature Layer** (Auth + Rate Limiting)
- Bearer token authentication
- Per-key rate limiting
- Token bucket algorithm

**Polish Layer** (Production Ready) ✨
- YAML/TOML configuration files
- Environment variable support
- Prometheus metrics
- Security headers and CORS
- HTTPS/TLS support
- Request size limits and timeouts
- Input validation (SQL injection, XSS, path traversal)
- Graceful shutdown
- Enhanced health checks

## Quick Start

### Installation

```bash
# Install with production dependencies
pip install simply-mcp fastapi uvicorn prometheus-client pyyaml

# Optional: For TOML support on Python <3.11
pip install tomli
```

### Basic Production Server

```python
import asyncio
from simply_mcp import BuildMCPServer
from simply_mcp.core.http_config import HttpConfig
from simply_mcp.transports.http_transport import HttpTransport

# Create MCP server
mcp = BuildMCPServer(name="my-server", version="1.0.0")

# Register tools
@mcp.tool()
def echo(message: str) -> str:
    return message

# Load configuration
config = HttpConfig.from_file("config.yaml")

# Create transport with configuration
transport = HttpTransport(server=mcp, config=config)

# Run server
async def main():
    async with transport:
        await asyncio.Event().wait()

asyncio.run(main())
```

## Configuration

### Configuration Files

Create a `config.yaml` file:

```yaml
environment: production

server:
  host: "0.0.0.0"
  port: 8000
  graceful_timeout: 30

tls:
  enabled: true
  cert_file: "/etc/ssl/certs/server.crt"
  key_file: "/etc/ssl/private/server.key"

auth:
  enabled: true
  key_env_var: "MCP_API_KEY"

rate_limit:
  enabled: true
  default_limit: 1000
  window_seconds: 60

monitoring:
  prometheus_enabled: true
  prometheus_path: "/metrics"
  health_path: "/health"

security:
  security_headers: true
  hsts_enabled: true
  max_request_size: 10485760  # 10MB
  request_timeout: 30

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/mcp/server.log"
```

### Environment Variables

Override any configuration with environment variables:

```bash
# Server configuration
export MCP_HTTP_SERVER__HOST=0.0.0.0
export MCP_HTTP_SERVER__PORT=9000

# Auth configuration
export MCP_HTTP_AUTH__ENABLED=true
export MCP_HTTP_API_KEY=your-secret-key

# TLS configuration
export MCP_HTTP_TLS__ENABLED=true
export MCP_HTTP_TLS__CERT_FILE=/path/to/cert.pem
export MCP_HTTP_TLS__KEY_FILE=/path/to/key.pem

# Logging
export MCP_HTTP_LOGGING__LEVEL=DEBUG
```

Environment variables follow the pattern: `MCP_HTTP_<SECTION>__<KEY>=<VALUE>`

### Configuration Priority

Configuration is loaded in this order (highest to lowest):

1. **Environment variables** - Override everything
2. **Config file** - YAML or TOML
3. **Default values** - Sensible defaults

### Loading Configuration

```python
# From YAML file
config = HttpConfig.from_file("config.yaml")

# From TOML file
config = HttpConfig.from_file("config.toml")

# From environment variables
config = HttpConfig.from_env()

# From dictionary
config = HttpConfig.from_dict({
    "environment": "production",
    "server": {"port": 9000},
})

# Use defaults
config = HttpConfig()
```

## Security

### Authentication

Enable authentication with API keys:

```yaml
auth:
  enabled: true
  key_env_var: "MCP_API_KEY"
```

Set the API key:

```bash
export MCP_API_KEY=your-secret-key-here
```

Make authenticated requests:

```bash
curl -H "Authorization: Bearer your-secret-key-here" \
     https://your-server.com/tools/echo \
     -d '{"message":"Hello"}'
```

### Security Headers

The following security headers are automatically added:

- **Strict-Transport-Security**: HSTS for HTTPS enforcement
- **X-Content-Type-Options**: Prevents MIME sniffing
- **X-Frame-Options**: Prevents clickjacking
- **X-XSS-Protection**: XSS filter
- **Referrer-Policy**: Controls referrer information

Configure in `config.yaml`:

```yaml
security:
  security_headers: true
  hsts_enabled: true
  hsts_max_age: 31536000  # 1 year
  content_type_nosniff: true
  xss_protection: true
  frame_options: "DENY"
```

### CORS

Configure Cross-Origin Resource Sharing:

```yaml
cors:
  enabled: true
  allow_origins:
    - "https://app.example.com"
    - "https://admin.example.com"
  allow_methods:
    - "GET"
    - "POST"
    - "PUT"
    - "DELETE"
  allow_headers:
    - "Content-Type"
    - "Authorization"
  allow_credentials: true
  max_age: 3600
```

### Request Limits

Protect against large requests and timeouts:

```yaml
security:
  max_request_size: 10485760  # 10MB
  request_timeout: 30  # seconds
```

### Input Validation

Automatic protection against:

- **SQL Injection**: Detects SQL patterns in requests
- **Path Traversal**: Blocks `../` and similar patterns
- **XSS**: Basic detection of script tags and event handlers

These protections are enabled by default when using the polish layer.

## Monitoring

### Prometheus Metrics

Enable metrics collection:

```yaml
monitoring:
  prometheus_enabled: true
  prometheus_path: "/metrics"
```

Access metrics:

```bash
curl http://your-server.com/metrics
```

#### Available Metrics

**Request Metrics:**
- `http_requests_total` - Total request count by method, endpoint, status
- `http_request_duration_seconds` - Request latency histogram
- `http_request_size_bytes` - Request body size
- `http_response_size_bytes` - Response body size

**Authentication Metrics:**
- `http_auth_success_total` - Successful auth attempts
- `http_auth_failure_total` - Failed auth attempts by reason

**Rate Limiting Metrics:**
- `http_rate_limit_hit_total` - Rate limit checks (within limit)
- `http_rate_limit_exceeded_total` - Rate limit denials
- `http_rate_limit_remaining` - Remaining requests per key

**Tool Execution Metrics:**
- `http_tool_execution_total` - Tool execution count by status
- `http_tool_execution_duration_seconds` - Tool execution latency
- `http_tool_execution_errors_total` - Tool errors by type

**System Metrics:**
- `http_active_connections` - Current active connections
- `http_request_queue_size` - Requests in queue

### Health Checks

Enhanced health endpoint with component status:

```bash
curl http://your-server.com/health
```

Response:

```json
{
  "status": "healthy",
  "server": "my-server",
  "version": "1.0.0",
  "timestamp": 1234567890.123,
  "components": {
    "server": {
      "status": "up",
      "host": "0.0.0.0",
      "port": 8000,
      "environment": "production"
    },
    "authentication": {
      "status": "enabled",
      "keys_count": 3
    },
    "rate_limiting": {
      "status": "enabled",
      "strategy": "token_bucket"
    },
    "metrics": {
      "status": "enabled",
      "endpoint": "/metrics"
    },
    "security": {
      "status": "enabled",
      "hsts": true,
      "cors": true
    }
  }
}
```

### Structured Logging

Enable JSON logging for log aggregation:

```yaml
logging:
  level: "INFO"
  format: "json"
  file: "/var/log/mcp/server.log"
  structured: true
```

Log format:

```json
{
  "timestamp": "2024-10-16T12:00:00.000Z",
  "level": "INFO",
  "logger": "simply_mcp.transports.http_transport",
  "message": "Request completed",
  "correlation_id": "abc-123-def",
  "context": {
    "method": "POST",
    "path": "/tools/echo",
    "status": 200,
    "duration": 0.052
  }
}
```

## TLS/HTTPS Setup

### Generate Self-Signed Certificate (Development)

```bash
# Generate private key and certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

### Configure TLS

```yaml
tls:
  enabled: true
  cert_file: "/path/to/cert.pem"
  key_file: "/path/to/key.pem"
  ca_file: "/path/to/ca.pem"  # Optional
  verify_client: false  # Client certificate verification
```

### Let's Encrypt (Production)

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d your-domain.com

# Certificate files will be in:
# /etc/letsencrypt/live/your-domain.com/fullchain.pem
# /etc/letsencrypt/live/your-domain.com/privkey.pem
```

Configure:

```yaml
tls:
  enabled: true
  cert_file: "/etc/letsencrypt/live/your-domain.com/fullchain.pem"
  key_file: "/etc/letsencrypt/live/your-domain.com/privkey.pem"
```

### Auto-Renewal

Add to crontab:

```bash
0 0 1 * * certbot renew --quiet --deploy-hook "systemctl restart mcp-server"
```

## Performance Tuning

### Worker Configuration

```yaml
server:
  workers: 4  # Number of worker processes (0 = auto)
  worker_connections: 1000  # Max connections per worker
  backlog: 2048  # Socket backlog size
  keepalive: 5  # Connection keepalive (seconds)
```

### Connection Limits

```yaml
security:
  max_request_size: 10485760  # 10MB
  request_timeout: 30  # seconds
```

### Rate Limiting

```yaml
rate_limit:
  enabled: true
  default_limit: 1000  # requests per window
  window_seconds: 60  # window size
  strategy: "token_bucket"
```

### Graceful Shutdown

```yaml
server:
  graceful_timeout: 30  # Allow 30s for connections to drain
```

## Deployment

### Docker

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "server.py"]
```

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_HTTP_SERVER__HOST=0.0.0.0
      - MCP_HTTP_SERVER__PORT=8000
      - MCP_HTTP_AUTH__ENABLED=true
      - MCP_HTTP_API_KEY=${MCP_API_KEY}
    volumes:
      - ./config.yaml:/app/config.yaml:ro
      - ./logs:/var/log/mcp
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
```

### Systemd Service

**/etc/systemd/system/mcp-server.service:**

```ini
[Unit]
Description=MCP HTTP Server
After=network.target

[Service]
Type=simple
User=mcp
Group=mcp
WorkingDirectory=/opt/mcp-server
ExecStart=/opt/mcp-server/venv/bin/python server.py
Restart=always
RestartSec=10

# Environment
Environment="MCP_HTTP_SERVER__HOST=0.0.0.0"
Environment="MCP_HTTP_SERVER__PORT=8000"
EnvironmentFile=/etc/mcp/server.env

# Logging
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/mcp

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable mcp-server
sudo systemctl start mcp-server
sudo systemctl status mcp-server
```

### Kubernetes

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: MCP_HTTP_SERVER__HOST
          value: "0.0.0.0"
        - name: MCP_HTTP_SERVER__PORT
          value: "8000"
        - name: MCP_HTTP_API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: api-key
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Nginx Reverse Proxy

**/etc/nginx/sites-available/mcp-server:**

```nginx
upstream mcp_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name your-domain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;

    # Proxy Configuration
    location / {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://mcp_backend/health;
        access_log off;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;  # Internal network
        deny all;
        proxy_pass http://mcp_backend/metrics;
    }
}
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
export MCP_HTTP_SERVER__PORT=9000
```

**2. Permission Denied (Port <1024)**

```bash
# Use port >=1024 or run as root (not recommended)
export MCP_HTTP_SERVER__PORT=8000

# Or use systemd socket activation
# Or use reverse proxy (recommended)
```

**3. TLS Certificate Errors**

```bash
# Verify certificate
openssl x509 -in cert.pem -text -noout

# Check certificate and key match
openssl x509 -noout -modulus -in cert.pem | openssl md5
openssl rsa -noout -modulus -in key.pem | openssl md5
```

**4. High Memory Usage**

```yaml
# Reduce workers
server:
  workers: 2  # Instead of 4

# Reduce connections
server:
  worker_connections: 500  # Instead of 1000
```

**5. Slow Requests**

```yaml
# Increase timeout
security:
  request_timeout: 60  # Instead of 30

# Check tool execution metrics
# curl http://localhost:8000/metrics | grep tool_execution_duration
```

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
  format: "text"
  enable_console: true
```

Or via environment:

```bash
export MCP_HTTP_LOGGING__LEVEL=DEBUG
```

### Logs

Check logs:

```bash
# Systemd
journalctl -u mcp-server -f

# Docker
docker logs -f mcp-server

# File
tail -f /var/log/mcp/server.log
```

### Metrics Analysis

Check metrics for issues:

```bash
# Get all metrics
curl http://localhost:8000/metrics

# Check error rate
curl http://localhost:8000/metrics | grep http_requests_total | grep status=\"500\"

# Check auth failures
curl http://localhost:8000/metrics | grep http_auth_failure_total

# Check rate limits
curl http://localhost:8000/metrics | grep http_rate_limit_exceeded_total
```

## Best Practices

### Security

1. **Always use HTTPS in production**
2. **Enable authentication** for sensitive operations
3. **Use strong API keys** (32+ characters, random)
4. **Rotate keys regularly**
5. **Enable rate limiting** to prevent abuse
6. **Keep dependencies updated**
7. **Use security headers**
8. **Restrict metrics endpoint** to internal network

### Monitoring

1. **Set up Prometheus scraping**
2. **Configure alerting** for errors and high latency
3. **Monitor health endpoint** with external service
4. **Enable structured logging**
5. **Set up log aggregation** (ELK, Loki, etc.)
6. **Track key metrics**:
   - Request rate and latency
   - Error rate
   - Auth failure rate
   - Rate limit hits

### Performance

1. **Use appropriate worker count** (2-4 per CPU core)
2. **Enable connection keepalive**
3. **Use reverse proxy** for SSL termination
4. **Configure request limits** appropriately
5. **Enable compression** at reverse proxy
6. **Use connection pooling**
7. **Cache static responses** if applicable

### Reliability

1. **Enable graceful shutdown** with appropriate timeout
2. **Implement health checks** at multiple levels
3. **Use automatic restarts** (systemd, k8s)
4. **Set resource limits** (memory, CPU)
5. **Configure backups** for critical data
6. **Test disaster recovery** procedures

## Support

For issues or questions:

- **GitHub Issues**: https://github.com/Clockwork-Innovations/simply-mcp-py/issues
- **Documentation**: https://simply-mcp-py.readthedocs.io
- **Examples**: `demo/gemini/` directory

---

**Version**: 1.0.0
**Last Updated**: October 2024
**Maintained by**: Clockwork Innovations
