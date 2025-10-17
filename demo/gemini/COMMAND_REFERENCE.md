# Gemini MCP Server - Command Reference

## Quick Commands

### Setup Commands

```bash
# Set your API key
export GEMINI_API_KEY="your-key-here"

# Option 1: Using claude mcp add (Recommended)
claude mcp add --transport stdio gemini \
  --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -- python /path/to/demo/gemini/server.py

# Option 2: Using simply-mcp run
simply-mcp run /path/to/demo/gemini/server.py

# Option 3: Using simply-mcp dev (with hot-reload)
simply-mcp dev /path/to/demo/gemini/server.py
```

### Management Commands (Claude MCP)

```bash
# List all configured servers
claude mcp list

# Get details about a specific server
claude mcp get gemini

# Remove a server
claude mcp remove gemini

# Show configuration
claude mcp config show
```

### Server Commands (simply-mcp)

```bash
# Run with stdio transport (default, for Claude Code)
simply-mcp run demo/gemini/server.py

# Run with HTTP transport (for API access)
simply-mcp run demo/gemini/server.py \
  --transport http \
  --port 3000 \
  --host 0.0.0.0

# Run with SSE transport
simply-mcp run demo/gemini/server.py \
  --transport sse \
  --port 3000

# Run in development mode (auto-reload)
simply-mcp dev demo/gemini/server.py

# Build a .pyz package
simply-mcp build demo/gemini/server.py -o gemini-server.pyz

# List available servers in a module
simply-mcp list demo/gemini/server.py
```

### Testing Commands

```bash
# Test if server starts
python demo/gemini/server.py

# Test with timeout (useful for CI/CD)
timeout 5 python demo/gemini/server.py

# Check Python has dependencies
python -c "import google.genai; print('✓ google-genai installed')"

# Run a simple test chat
python -c "
from demo.gemini.server import start_chat, send_message
result = start_chat('test', 'Hello')
print(result)
"
```

## Advanced Usage

### Using Environment Variables

```bash
# Set multiple environment variables
export GEMINI_API_KEY="your-key"
export GEMINI_DEFAULT_MODEL="gemini-2.5-flash"
export GEMINI_FILE_WARNING_HOURS="24"

# Add to Claude Code with all env vars
claude mcp add --transport stdio gemini \
  --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
  --env GEMINI_DEFAULT_MODEL="${GEMINI_DEFAULT_MODEL}" \
  -- python /path/to/demo/gemini/server.py
```

### Using .env File

```bash
# Create .env file in demo/gemini/
cat > demo/gemini/.env << 'EOF'
GEMINI_API_KEY=your-key-here
GEMINI_DEFAULT_MODEL=gemini-2.5-flash
EOF

# Server will automatically load this
python demo/gemini/server.py
```

### Using Configuration File

```bash
# Create config.toml
cat > demo/gemini/config.toml << 'EOF'
[gemini]
api_key = "your-key-here"
default_model = "gemini-2.5-flash"
file_warning_hours = 24
max_file_size = 2147483648
EOF

# Run with config
simply-mcp run demo/gemini/server.py --config demo/gemini/config.toml
```

### HTTP Server (For API Access)

```bash
# Start HTTP server
simply-mcp run demo/gemini/server.py --transport http --port 3000

# Test the API
curl http://localhost:3000/

# Call a tool via JSON-RPC
curl -X POST http://localhost:3000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "list_files",
      "arguments": {}
    }
  }'
```

## Installation Methods

### Method 1: Direct Python (No Dependencies)

```bash
# Requires: python3.10+ (system)
python demo/gemini/server.py
```

### Method 2: With Virtual Environment

```bash
# Requires: venv with dependencies
source /path/to/venv/bin/activate
python demo/gemini/server.py
```

### Method 3: With simply-mcp CLI

```bash
# Requires: simply-mcp installed globally
pip install simply-mcp
simply-mcp run demo/gemini/server.py
```

### Method 4: Using npx (Requires NPM and simply-mcp on npm)

```bash
# Future: Once simply-mcp is published to npm
npx simply-mcp run demo/gemini/server.py

# Or with options
npx simply-mcp run demo/gemini/server.py \
  --transport http \
  --port 3000
```

### Method 5: Using .pyz Package

```bash
# Build package
simply-mcp build demo/gemini/server.py -o gemini-server.pyz

# Run package
python gemini-server.pyz

# Or with simply-mcp
simply-mcp run gemini-server.pyz --transport http --port 3000
```

### Method 6: Using .mcpb Bundle

```bash
# Create bundle (see SETUP_GUIDE.md)
mcpb pack gemini-mcp -o gemini-server.mcpb

# Install in Claude Desktop
# - Double-click gemini-server.mcpb
# OR
claude mcp install gemini-server.mcpb
```

## Configuration Locations

### By Scope

**Project Scope** (Version controlled):
```
<project>/.mcp.json
```

**User Scope** (User settings):
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Local Scope** (Override for current session):
```
<project>/.mcp-local.json
```

## Common Workflows

### Development Workflow

```bash
# 1. Terminal 1: Run server in development mode
cd /mnt/Shared/cs-projects/simply-mcp-py
simply-mcp dev demo/gemini/server.py

# 2. Terminal 2: Run Claude Code
# (Claude Code will auto-connect via configuration)

# 3. Make changes to server.py
# 4. Server auto-reloads (dev mode watches for changes)
```

### Testing Workflow

```bash
# 1. Build a package
simply-mcp build demo/gemini/server.py -o gemini-server.pyz

# 2. Test the package
simply-mcp run gemini-server.pyz

# 3. If working, distribute or deploy
```

### Team Collaboration Workflow

```bash
# 1. Create bundle
mcpb pack demo/gemini -o gemini-server.mcpb

# 2. Share gemini-server.mcpb with team

# 3. Team members double-click to install
# OR
claude mcp install gemini-server.mcpb

# 4. Each team member sets their own API key when prompted
```

### CI/CD Workflow

```bash
# GitHub Actions example
- name: Validate Gemini Server
  run: |
    pip install simply-mcp google-genai
    timeout 5 python demo/gemini/server.py || true
    simply-mcp list demo/gemini/server.py

- name: Build Package
  run: |
    simply-mcp build demo/gemini/server.py -o gemini-server.pyz
```

## Environment Variables Quick Reference

| Variable | Type | Default | Required |
|----------|------|---------|----------|
| `GEMINI_API_KEY` | string | - | Yes |
| `GEMINI_DEFAULT_MODEL` | string | gemini-2.5-flash | No |
| `GEMINI_FILE_WARNING_HOURS` | int | 24 | No |
| `GEMINI_MAX_FILE_SIZE` | int | 2GB | No |
| `PYTHONUNBUFFERED` | 1 | - | No |

## Useful Aliases

Add these to your `.bashrc` or `.zshrc`:

```bash
# Run server
alias gemini-start="simply-mcp run $(pwd)/demo/gemini/server.py"

# Run in dev mode
alias gemini-dev="simply-mcp dev $(pwd)/demo/gemini/server.py"

# Add to Claude Code
alias gemini-add="claude mcp add --transport stdio gemini --env GEMINI_API_KEY=\${GEMINI_API_KEY} -- python $(pwd)/demo/gemini/server.py"

# List Claude MCP servers
alias mcp-list="claude mcp list"

# Remove from Claude Code
alias gemini-remove="claude mcp remove gemini"
```

## Troubleshooting Commands

```bash
# Check Python version
python --version

# Check dependencies
pip list | grep -E "google-genai|simply-mcp|pydantic"

# Check API key
echo $GEMINI_API_KEY

# Test connectivity
curl -X POST https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent \
  -H "x-goog-api-key: ${GEMINI_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'

# View Claude MCP config
cat ~/.config/Claude/claude_desktop_config.json | python -m json.tool

# Check if server is running (HTTP mode)
curl http://localhost:3000/

# Check logs (if server is in background)
tail -f /tmp/gemini-mcp-server.log
```

## Useful Links

- API Key: https://aistudio.google.com/app/apikey
- Gemini Models: https://ai.google.dev/models
- Simply-MCP Docs: https://github.com/Clockwork-Innovations/simply-mcp-py
- MCP Protocol: https://modelcontextprotocol.io
- Claude Code: https://docs.claude.com/

---

**Need help?** See [SETUP_GUIDE.md](./SETUP_GUIDE.md) or [QUICK_START.md](./QUICK_START.md)
