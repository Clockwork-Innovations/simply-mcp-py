# Gemini MCP Server - Quick Start Guide

## 3-Minute Setup

### Option 1: Using `claude mcp add` (Recommended)

```bash
# 1. Set your API key
export GEMINI_API_KEY="your-api-key-from-https://aistudio.google.com/app/apikey"

# 2. Add the server to Claude Code
claude mcp add --transport stdio gemini \
  --env GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -- python /path/to/demo/gemini/server.py

# 3. Restart Claude Code completely
# (Quit and reopen - changes only take effect on restart)

# 4. Done! You'll see Gemini tools in Claude Code
```

### Option 2: Using Configuration File

**macOS/Linux:**
```bash
cat > ~/.config/Claude/claude_desktop_config.json << 'EOF'
{
  "mcpServers": {
    "gemini-server": {
      "command": "python",
      "args": ["/path/to/demo/gemini/server.py"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
EOF
```

**Windows:**
```powershell
# Edit: %APPDATA%\Claude\claude_desktop_config.json
# Add the configuration above
```

Then restart Claude Code.

### Option 3: Using simply-mcp CLI

If you have `simply-mcp` installed globally:

```bash
# Run directly
simply-mcp run /path/to/demo/gemini/server.py

# Or with HTTP transport
simply-mcp run /path/to/demo/gemini/server.py --transport http --port 3000
```

## Getting Your API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API key"
3. Copy the key
4. Paste it in the command above

## What You Get

Once configured, you'll have access to:

### 6 Tools
- **upload_file** - Upload files to Gemini
- **generate_content** - Generate content with files
- **start_chat** - Start a chat session
- **send_message** - Send chat messages
- **list_files** - List uploaded files
- **delete_file** - Delete files

### 3 Prompts
- **analyze_media** - Analyze images, videos, audio, documents
- **document_qa** - Question answering on documents
- **multimodal_analysis** - Compare or synthesize multiple files

### 2 Resources
- **chat-history** - Get chat session info
- **file-info** - Get file metadata

## Example Usage

### Simple Chat
```
You: Chat with Gemini about something
Claude Code: [Uses start_chat and send_message tools]
Gemini: Responds back
```

### Upload and Analyze
```
You: Analyze this image using Gemini
Claude Code: [Uploads image, generates analysis]
```

## Troubleshooting

### Tools Not Showing Up

1. **Did you restart Claude Code?** - Configuration only loads on app start
   - Quit Claude Code completely (Command+Q on Mac, not just close window)
   - Reopen Claude Code
   - Tools should now appear

2. **Check API Key** - Make sure GEMINI_API_KEY is set
   ```bash
   echo $GEMINI_API_KEY
   ```

3. **Verify Configuration** - Check the config file exists and has valid JSON
   ```bash
   # macOS/Linux
   cat ~/.config/Claude/claude_desktop_config.json
   python -m json.tool < ~/.config/Claude/claude_desktop_config.json
   ```

### "google-genai SDK not available"

Make sure you have dependencies installed:
```bash
pip install google-genai python-dotenv simply-mcp
```

Or use the virtual environment where they're installed:
```bash
/path/to/venv/bin/python /path/to/demo/gemini/server.py
```

## Advanced: Using `.mcpb` Bundle

For one-click installation in Claude Desktop:

```bash
# 1. Create bundle structure
mkdir -p gemini-mcp/server
cp demo/gemini/server.py gemini-mcp/server/
cp demo/gemini/requirements.txt gemini-mcp/server/
cp demo/gemini/manifest.json gemini-mcp/

# 2. Pack it
pip install mcpb
mcpb pack gemini-mcp -o gemini-server.mcpb

# 3. Users can now double-click gemini-server.mcpb to install
# Or: claude mcp install gemini-server.mcpb
```

## Next Steps

1. ✅ Set GEMINI_API_KEY environment variable
2. ✅ Run one of the setup commands above
3. ✅ Restart Claude Code completely
4. ✅ Try using the tools in your Claude Code chat
5. ✅ Upload an image and ask Gemini to analyze it
6. ✅ Start a multi-turn conversation with Gemini

## Documentation

For more details, see:
- [SETUP_GUIDE.md](./SETUP_GUIDE.md) - Comprehensive setup guide
- [README.md](./README.md) - Full feature documentation
- [manifest.json](./manifest.json) - Server metadata and configuration

## Get Help

- Check error messages in the console
- Verify API key is valid at https://aistudio.google.com/app/apikey
- See SETUP_GUIDE.md for detailed troubleshooting
- Open an issue at https://github.com/Clockwork-Innovations/simply-mcp-py/issues

---

**That's it!** You should now be able to use Gemini tools in Claude Code. Enjoy! 🚀
