# Gemini MCP Server - UVX Setup Guide

## Overview

This guide explains how to use `uvx` (uv's execute command) to run the Gemini MCP Server. This approach automatically downloads, caches, and manages all dependencies - no virtual environment setup needed!

### Why Use `uvx`?

✅ **Automatic dependency management** - Downloads and caches automatically
✅ **No venv needed** - Works out of the box
✅ **Cross-platform** - Same command works on macOS, Linux, Windows
✅ **Cleaner setup** - Much simpler configuration
✅ **Faster updates** - Dependencies updated via uv, not manual venv maintenance

---

## Requirements

- `uv` installed (fast Python package manager)
  - Installation: https://docs.astral.sh/uv/getting-started/installation/
  - Test: `uv --version`

---

## Setup Methods

### Method 1: Using `claude mcp add` with `uvx`

```bash
# 1. Set your API key
export GEMINI_API_KEY="your-api-key-from-https://aistudio.google.com/app/apikey"

# 2. Add server using uvx
cd /mnt/Shared/cs-projects/simply-mcp-py

claude mcp add --transport stdio gemini \
  --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -- uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py

# 3. Restart Claude Code completely
# Quit entirely and reopen
```

**Result:** Tools will be available in Claude Code with automatic dependency management!

---

### Method 2: Manual Configuration File

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "uvx",
      "args": [
        "simply-mcp",
        "run",
        "/mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Then restart Claude Code completely.

---

### Method 3: Direct Command Line

Run the server directly:

```bash
# Basic run
uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py

# With HTTP transport
uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py \
  --transport http \
  --port 3000

# Development mode (auto-reload)
uvx simply-mcp dev /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py
```

**What happens:**
1. `uvx` detects `simply-mcp` is needed
2. Downloads it from PyPI (first run only)
3. Caches it locally (~/.cache/uv/)
4. Runs the command with all dependencies available
5. Subsequent runs are instant (cached)

---

## How `uvx` Works

### First Run (Downloads)
```bash
$ uvx simply-mcp run server.py
# uv downloads simply-mcp and dependencies (takes ~30 seconds)
# Dependencies are cached in ~/.cache/uv/
# Server starts
```

### Subsequent Runs (Instant)
```bash
$ uvx simply-mcp run server.py
# Uses cached dependencies (instant)
# Server starts immediately
```

### Cache Location
```bash
~/.cache/uv/
```

---

## Dependency Management

### What Gets Cached

When you run `uvx simply-mcp run`, it automatically caches:

- `simply-mcp` - The MCP framework
- `google-genai` - Gemini API SDK
- `pydantic` - Data validation
- All other transitive dependencies

### Updating Dependencies

To update to the latest versions:

```bash
# Update all cached packages
uv cache prune

# Or run with --upgrade flag
uvx --upgrade-package simply-mcp simply-mcp run server.py
```

### Version Pinning

To use a specific version:

```bash
uvx simply-mcp==0.1.0 simply-mcp run server.py
```

---

## Environment Variables

### Setting API Key

**Option 1: Environment Variable**
```bash
export GEMINI_API_KEY="your-key"
uvx simply-mcp run /path/to/server.py
```

**Option 2: .env File**
```bash
# Create .env in demo/gemini/
echo "GEMINI_API_KEY=your-key" > demo/gemini/.env

# Then run
uvx simply-mcp run /path/to/server.py
# The server automatically loads .env
```

**Option 3: MCP Configuration**
```json
"env": {
  "GEMINI_API_KEY": "your-key"
}
```

---

## Advanced Usage

### Running Multiple Instances

Each instance has its own dependency cache:

```bash
# Terminal 1: Production
uvx simply-mcp run /path/to/server.py

# Terminal 2: Development (different cache if needed)
uvx --no-cache simply-mcp run /path/to/server.py
```

### Different Python Versions

Specify Python version for the environment:

```bash
# Use Python 3.11
uvx --python 3.11 simply-mcp run server.py

# Use Python 3.10
uvx --python 3.10 simply-mcp run server.py
```

### Offline Mode

Once cached, you can run offline:

```bash
# This will use cached dependencies even without internet
uvx --offline simply-mcp run server.py
```

---

## Troubleshooting

### Issue: "uvx command not found"

**Solution:** Install `uv`
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

---

### Issue: "simply-mcp not found in PyPI"

**Solution:** Make sure you're online for first run
```bash
# First run downloads from PyPI
uvx simply-mcp run server.py

# Verify it cached
ls ~/.cache/uv/
```

---

### Issue: "google-genai SDK not available"

**Solution:** This happens automatically with uvx
```bash
# uvx handles this - it downloads all dependencies
# If error persists, clear cache and retry
uv cache prune
uvx simply-mcp run server.py
```

---

### Issue: "Permission denied" on uvx

**Solution:** Ensure uvx is executable
```bash
# Check location
which uvx

# Make executable if needed
chmod +x ~/.local/bin/uv
```

---

### Issue: Slow on first run

**Normal behavior!** First run downloads everything:
- `simply-mcp` package
- `google-genai` SDK
- All dependencies
- This is cached for future runs

Subsequent runs are instant.

---

## Comparison: Before vs After

### Before (Virtual Environment)
```bash
# Setup (one time)
python -m venv venv
source venv/bin/activate
pip install simply-mcp google-genai python-dotenv

# Configure Claude Code
# command: /path/to/venv/bin/python
# args: [/path/to/server.py]

# Run
simply-mcp run server.py
```

### After (UVX)
```bash
# Setup (none needed!)

# Configure Claude Code
# command: uvx
# args: [simply-mcp, run, /path/to/server.py]

# Run
uvx simply-mcp run server.py
```

**Much simpler!** ✨

---

## Migration from Venv to UVX

If you currently use the venv setup:

### Step 1: Update Claude Code Config
Edit `~/.config/Claude/claude_desktop_config.json`:

Change from:
```json
{
  "command": "/path/to/venv/bin/python",
  "args": ["/path/to/server.py"],
  "env": {"PYTHONUNBUFFERED": "1"}
}
```

To:
```json
{
  "command": "uvx",
  "args": ["simply-mcp", "run", "/path/to/server.py"],
  "env": {"PYTHONUNBUFFERED": "1"}
}
```

### Step 2: Restart Claude Code
Quit completely and reopen.

### Step 3: Test
Verify tools appear in Claude Code.

### Step 4: (Optional) Clean Up
```bash
# You can now delete the venv if you want
rm -rf venv

# The uvx cache takes much less space anyway
```

---

## Performance

### First Run
- Download time: ~30 seconds (one-time)
- Startup: ~5 seconds

### Subsequent Runs
- Cached startup: <1 second
- No download overhead

### Cache Size
- Typical cache: ~200MB (for all dependencies)
- Much smaller than multiple venvs

---

## Best Practices

✅ **Do:**
- Keep `uvx` updated: `uv self update`
- Periodically clean cache: `uv cache prune`
- Use absolute paths in Claude Code config
- Set `PYTHONUNBUFFERED=1` for proper output
- Use `.env` files for sensitive data

❌ **Don't:**
- Hardcode API keys in config files (use env vars or .env)
- Modify cached dependencies manually
- Mix uvx with system Python packages
- Use relative paths in Claude Code config

---

## Useful Commands

```bash
# Show uv version
uv --version

# Show uv cache info
uv cache dir

# Clean all cache
uv cache prune

# Run with verbose output
uvx --verbose simply-mcp run server.py

# Check what will be downloaded (dry run)
uv pip compile demo/gemini/requirements.txt

# Use specific Python version
uvx --python 3.11 simply-mcp run server.py

# Update packages
uv cache prune  # Clear cache, will re-download latest
uvx simply-mcp run server.py  # Re-downloads latest versions
```

---

## Current Setup Status

✅ **Claude Code Configuration Updated**

Your Claude Code is now configured to use `uvx`:

```json
{
  "command": "uvx",
  "args": ["simply-mcp", "run", "/mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py"]
}
```

**Next Steps:**
1. ✅ Configuration is done
2. Quit Claude Code completely
3. Reopen Claude Code
4. Set `GEMINI_API_KEY` environment variable (if not already set)
5. Tools should appear automatically

---

## See Also

- [QUICK_START.md](./QUICK_START.md) - 3-minute setup guide
- [SETUP_COMPLETE.md](./SETUP_COMPLETE.md) - Comprehensive guide
- [COMMAND_REFERENCE.md](./COMMAND_REFERENCE.md) - All commands
- [uv Documentation](https://docs.astral.sh/uv/) - Official uv docs

---

**Last Updated:** 2025-10-16
**Status:** ✅ Configuration updated and ready to use
**Approach:** UVX-based (automatic dependency management)
