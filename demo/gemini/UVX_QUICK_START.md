# Gemini MCP Server - UVX Quick Start (30 seconds)

## Setup (Choose One)

### Option 1: Using `claude mcp add` with `uvx` (Easiest ⭐)

```bash
# 1. Set API key
export GEMINI_API_KEY="your-api-key-from-https://aistudio.google.com/app/apikey"

# 2. Add server using uvx
cd /mnt/Shared/cs-projects/simply-mcp-py

claude mcp add --transport stdio gemini \
  --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -- uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py

# 3. Restart Claude Code completely
# Quit entirely and reopen (important!)
```

**Done! Your Claude Code now has Gemini tools with automatic dependency management.**

---

### Option 2: Pre-Configured (Already Set Up! 🎉)

If the configuration is already in place, just:

```bash
# 1. Set API key
export GEMINI_API_KEY="your-api-key-from-https://aistudio.google.com/app/apikey"

# 2. Restart Claude Code completely
# Quit entirely and reopen
```

**Tools should appear automatically!**

---

### Option 3: Direct Command Line Testing

Test it works without Claude Code:

```bash
export GEMINI_API_KEY="your-api-key"
cd /mnt/Shared/cs-projects/simply-mcp-py

uvx simply-mcp run demo/gemini/server.py
```

On first run, it will:
1. Download `simply-mcp` (~30 seconds)
2. Download dependencies (`google-genai`, etc.)
3. Cache everything locally
4. Start the server

On subsequent runs, it's instant (uses cache).

---

## What's Happening

```
uvx simply-mcp run /path/to/server.py
 ↓
uvx downloads simply-mcp from PyPI (first run only)
 ↓
uvx downloads dependencies (first run only)
 ↓
All cached in ~/.cache/uv/
 ↓
simply-mcp run starts server
 ↓
Server ready with Gemini tools!
```

---

## Key Advantages

✅ **No Virtual Environment Setup Needed**
- Just run the command
- Dependencies handled automatically

✅ **Automatic Updates**
- uv keeps packages current
- Clean cache, get latest versions

✅ **Cross-Platform**
- Same command on macOS, Linux, Windows
- Works with existing `claude mcp add` command

✅ **Smaller Footprint**
- Shared cache (~200MB total)
- No duplicate venvs

✅ **Instant on Cached Runs**
- First run: ~30 seconds (download)
- Subsequent runs: <1 second

---

## Verify It Works

After restarting Claude Code:

1. ✅ Look for tools in available tools list
2. ✅ Should see: `mcp__gemini-server__start_chat`, `mcp__gemini-server__upload_file`, etc.
3. ✅ Try using one of the tools
4. ✅ If it works, you're all set!

---

## Troubleshooting

**Tools not showing?**
- ✅ Did you quit Claude Code completely? (not just close window)
- ✅ Did you reopen Claude Code?
- ✅ Is `uvx` installed? (`which uvx`)
- ✅ Is API key set? (`echo $GEMINI_API_KEY`)

**"uvx: command not found"?**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify
uv --version
```

**Server won't start?**
```bash
# Test manually
uvx simply-mcp run demo/gemini/server.py

# Should download and start (takes ~30 seconds first time)
```

---

## Next Steps

1. ✅ Run setup command above
2. ✅ Restart Claude Code
3. ✅ Check for Gemini tools
4. ✅ Try using them!

For more details, see [UVX_SETUP.md](./UVX_SETUP.md)

---

**Status:** ✅ Ready to use
**Setup Time:** 30 seconds + API key
**Dependencies:** Automatic (via uvx)
