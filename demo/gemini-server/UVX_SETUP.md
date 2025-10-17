# Using Gemini MCP Server with UVX

This guide shows how to set up and use the Gemini MCP Server with `uvx` and Claude Code.

## What is UVX?

`uvx` (uv execute) is part of the `uv` Python package manager. It:
- Downloads and caches Python packages
- Runs packages without requiring pre-installation
- Manages dependencies automatically
- Provides isolated execution environments

## Prerequisites

1. **Install uv**: https://docs.astral.sh/uv/getting-started/installation/
2. **Get a Gemini API Key**: https://aistudio.google.com/app/apikey
3. **Claude Code Desktop** (for MCP integration)

## Step 1: Verify UVX is Installed

```bash
uvx --version
# Expected output: uv 0.x.x
```

## Step 2: Test Direct Execution with UVX

```bash
# This will download and run the package without installation
export GEMINI_API_KEY="your-api-key-here"
uvx gemini-mcp-server
```

If you see the server startup message, it's working!

```
======================================================================
Gemini MCP Server - Feature Layer
======================================================================

Server: gemini-server v0.1.0
Transport: stdio

Available Tools:
  1. upload_file      - Upload files to Gemini Files API
  2. generate_content - Generate content with optional file context
  3. start_chat       - Start chat sessions with initial messages
  4. send_message     - Send messages to continue chat sessions
  5. list_files       - List all uploaded files with metadata
  6. delete_file      - Delete files from Gemini Files API

Features:
  - Gemini SDK: Available
  - API Key: Configured
  - File Registry: Enabled
  - Chat Sessions: Enabled
  - File Management: Enabled

======================================================================

Server is ready and listening on stdio...
```

Press `Ctrl+C` to stop.

## Step 3: Configure Claude Code

Edit `~/.config/Claude/claude_desktop_config.json`:

### Option A: Using UVX (Recommended)

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "uvx",
      "args": ["gemini-mcp-server"],
      "env": {
        "GEMINI_API_KEY": "AIzaSy..."
      }
    }
  }
}
```

**Advantages:**
- ✅ No local installation required
- ✅ Always uses latest version from PyPI
- ✅ Automatic caching
- ✅ Isolated environment

### Option B: Using Local Installation

If you prefer to install locally first:

```bash
pip install gemini-mcp-server
```

Then configure:

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "gemini-server",
      "env": {
        "GEMINI_API_KEY": "AIzaSy..."
      }
    }
  }
}
```

## Step 4: Use in Claude Code

After configuring and restarting Claude Code:

```
I need to work with Gemini's API. Can you help me analyze a file?
```

Claude Code will now have access to:
- **Tools**: upload_file, generate_content, start_chat, send_message, list_files, delete_file
- **Resources**: chat history, file information
- **Prompts**: media analysis, document Q&A, multimodal analysis

### Example: Upload and Analyze a Document

You can ask Claude Code:

```
Please analyze this PDF document: /path/to/document.pdf
```

Claude Code will:
1. Upload the file using the `upload_file` tool
2. Generate analysis using the `generate_content` tool with file context
3. Return the analysis results

## Caching and Performance

UVX automatically caches the downloaded package:

```bash
# First run: Downloads and caches (~50MB with dependencies)
uvx gemini-mcp-server
# Takes a few seconds

# Subsequent runs: Uses cached version
uvx gemini-mcp-server
# Starts immediately
```

Cache location: `~/.cache/uv/` (Unix-like systems)

## Troubleshooting

### "Command not found: uvx"

Install uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### "google-genai SDK not available"

The SDK should be automatically installed via uvx. If you see this error:

```bash
# Option 1: Reinstall the package
uvx --refresh gemini-mcp-server

# Option 2: Install locally
pip install gemini-mcp-server
```

### "GEMINI_API_KEY environment variable not set"

Set your API key:

```bash
# In claude_desktop_config.json
"env": {
  "GEMINI_API_KEY": "your-key-here"
}

# Or via environment variable
export GEMINI_API_KEY="your-key-here"
uvx gemini-mcp-server
```

### "404 Unauthorized. Please pass a valid API key"

- Verify your API key at https://aistudio.google.com/app/apikey
- Make sure the key is correctly set in the environment
- Check that the Gemini API is enabled in your Google project

## Updating to Newer Versions

When a new version is released:

```bash
# UVX automatically checks for updates
uvx --upgrade gemini-mcp-server

# Or force refresh
uvx --refresh gemini-mcp-server
```

For Claude Code, just restart the application and the new version will be used.

## Advanced: Using with Custom Python Versions

UVX respects Python version constraints:

```bash
# Specify Python version
uvx --python 3.11 gemini-mcp-server

# Use specific Python executable
uvx --python /usr/bin/python3.11 gemini-mcp-server
```

## Advanced: Local Development

If you're developing the package:

```bash
cd demo/gemini-server

# Install in editable mode
pip install -e .

# Run for testing
export GEMINI_API_KEY="your-key"
gemini-server
```

## Performance Notes

- **Memory**: ~150-200MB (including Python runtime and dependencies)
- **Startup time**: ~1-2 seconds (cached), ~5-10 seconds (fresh download)
- **Network**: Only downloads on first run or when updating

## What's Happening Behind the Scenes

When you run `uvx gemini-mcp-server`:

1. **Check Cache**: UVX looks for the package in `~/.cache/uv/`
2. **Download**: If not cached, downloads from PyPI (gemini-mcp-server package)
3. **Install Dependencies**: Downloads google-genai, mcp, pydantic, etc.
4. **Execute**: Runs the `gemini-server` entry point
5. **Listen**: Server listens on stdio for MCP protocol messages
6. **Cache**: Stores the package for future use

## Next Steps

- Read [README.md](./README.md) for detailed API documentation
- Check [PACKAGE_STRUCTURE.md](./PACKAGE_STRUCTURE.md) for technical details
- See [../](../) for the broader simply-mcp project

## Support

- **Issues**: https://github.com/Clockwork-Innovations/simply-mcp-py/issues
- **UVX Docs**: https://docs.astral.sh/uv/guides/tools/
- **MCP Docs**: https://modelcontextprotocol.io/
