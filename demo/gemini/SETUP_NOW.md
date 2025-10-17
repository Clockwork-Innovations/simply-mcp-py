# Gemini MCP Server - Setup NOW (Copy & Paste)

## ⚠️ SECURITY NOTE

The command below contains your API key. Keep it secret!
- Don't commit it to git
- Don't share it
- Regenerate if accidentally exposed

---

## 🚀 Setup Command (Copy & Paste)

### Step 1: Get Your API Key

Go to: **https://aistudio.google.com/app/apikey**

Create a new API key and copy it.

---

### Step 2: Run This Command

Replace `YOUR_API_KEY_HERE` with your actual key:

```bash
claude mcp add --transport stdio gemini-server \
  --env GEMINI_API_KEY=YOUR_API_KEY_HERE \
  -- uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py
```

**Example with real key (keep this SECRET!):**
```bash
claude mcp add --transport stdio gemini-server \
  --env GEMINI_API_KEY=AIzaSyB8fxcQqvILoDXsjIaJdqdrwZauE57UE \
  -- uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py
```

---

### Step 3: Restart Claude Code

1. **Quit Claude Code completely** (don't just close the window)
   - macOS: `Command + Q`
   - Windows: Close the application entirely
   - Linux: Close the window

2. **Wait 3 seconds**

3. **Reopen Claude Code**

---

### Step 4: Verify

After restarting Claude Code, you should see Gemini tools:
- ✅ `mcp__gemini-server__upload_file`
- ✅ `mcp__gemini-server__generate_content`
- ✅ `mcp__gemini-server__start_chat`
- ✅ `mcp__gemini-server__send_message`
- ✅ `mcp__gemini-server__list_files`
- ✅ `mcp__gemini-server__delete_file`

If you see these tools, **you're all set!** 🎉

---

## 📝 Command Breakdown

```
claude mcp add                    # Add MCP server to Claude Code
  --transport stdio              # Use stdio transport (for Claude Code)
  gemini-server                  # Name for this server
  --env GEMINI_API_KEY=...       # Pass API key as environment variable
  -- uvx                         # Use uvx to run command
  simply-mcp run                 # Run simply-mcp with run command
  /path/to/server.py             # Path to Gemini server
```

---

## 🔧 Alternative: Setting via Environment Variable

If you don't want the key in the command, set it first:

```bash
# 1. Set environment variable
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# 2. Run command without --env flag
claude mcp add --transport stdio gemini-server \
  -- uvx simply-mcp run /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py

# 3. Restart Claude Code
```

---

## 📁 Alternative: Configuration File

Instead of command line, you can edit `~/.config/Claude/claude_desktop_config.json`:

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
        "GEMINI_API_KEY": "YOUR_API_KEY_HERE"
      }
    }
  }
}
```

Then restart Claude Code.

---

## ✅ Verification Checklist

After setup:

- [ ] API key obtained from https://aistudio.google.com/app/apikey
- [ ] Command executed successfully
- [ ] Claude Code quit completely and restarted
- [ ] Gemini tools visible in Claude Code
- [ ] Can use `mcp__gemini-server__start_chat` or similar tools

---

## 🆘 Troubleshooting

### Tools not showing?

1. **Did you quit Claude Code completely?**
   - Just closing the window isn't enough
   - Use Quit/Exit command or keyboard shortcut
   - Wait 3 seconds
   - Reopen

2. **Check command executed**
   ```bash
   # Should show: Successfully added
   # If error, check your API key and path
   ```

3. **Verify uvx works**
   ```bash
   uvx simply-mcp --help
   ```

4. **Check Claude Code config**
   ```bash
   cat ~/.config/Claude/claude_desktop_config.json | python -m json.tool
   ```

### "uvx: command not found"?

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### API key error?

1. Check key is valid: https://aistudio.google.com/app/apikey
2. Regenerate if needed
3. Update command with new key
4. Re-run `claude mcp add` command

---

## 📚 Next Steps

Once set up:

1. **Try a simple chat**
   - Use `mcp__gemini-server__start_chat` tool
   - Start a conversation with Gemini

2. **Upload a file**
   - Use `mcp__gemini-server__upload_file` tool
   - Upload an image or document

3. **Analyze media**
   - Use the `analyze_media` prompt
   - Analyze images, videos, audio, or documents

4. **Multi-turn conversation**
   - Use `start_chat` to start
   - Use `send_message` for follow-ups

---

## 📖 For More Information

- **UVX Guide:** See `UVX_SETUP.md`
- **All Commands:** See `COMMAND_REFERENCE.md`
- **Troubleshooting:** See `SETUP_GUIDE.md`
- **Navigation:** See `INDEX.md`

---

## 🎉 You're All Set!

The Gemini MCP Server is now set up with automatic dependency management via uvx.

Key benefits:
- ✅ No venv needed
- ✅ Dependencies auto-managed
- ✅ First run: ~30 seconds (downloads)
- ✅ Subsequent: instant (cached)
- ✅ Zero maintenance

**Just set your API key and restart Claude Code!** 🚀

---

**Last Updated:** 2025-10-16
**Status:** ✅ Ready to use
**Setup Time:** 5 minutes (including API key retrieval)
