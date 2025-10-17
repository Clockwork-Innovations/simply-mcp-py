# Gemini MCP Server - Claude Code Setup Guide

## Overview

This guide explains how to properly configure the Gemini MCP Server to work with Claude Code's MCP interface.

## Prerequisites

✅ **Already Installed:**
- `google-genai` (Gemini API SDK)
- `simply-mcp` (MCP framework)
- `python-dotenv` (environment configuration)
- Python 3.10+ in virtual environment

✅ **Verified:**
- Server refactored to module-level functions (no introspection errors)
- All 6 tools registered and working
- Server starts successfully via venv Python

## Setup Steps

### Step 1: Verify Dependencies

The dependencies are already installed in the venv:

```bash
source /mnt/Shared/cs-projects/simply-mcp-py/venv/bin/activate
python -c "import google.genai; import simply_mcp; print('✓ All dependencies available')"
```

### Step 2: Configure Claude Code

Edit or create your Claude Code configuration file:

**File location:**
- **Linux:** `~/.config/Claude/claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

**Configuration content:**

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "/mnt/Shared/cs-projects/simply-mcp-py/venv/bin/python",
      "args": [
        "/mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Key configuration points:**
- **command:** Absolute path to Python in the virtual environment (ensures dependencies are available)
- **args:** Absolute path to the server script
- **env:** Environment variables (PYTHONUNBUFFERED prevents output buffering)
- The GEMINI_API_KEY is loaded from `.env` file by the server automatically

### Step 3: Restart Claude Code

⚠️ **IMPORTANT:** You must completely quit and restart Claude Code:
1. Quit Claude Code entirely (don't just close the window)
2. Wait a few seconds
3. Reopen Claude Code

Configuration changes only take effect on application startup.

### Step 4: Verify MCP Connection

After restarting Claude Code, you should see the Gemini server tools available. They will be named like:
- `mcp__gemini-server__upload_file`
- `mcp__gemini-server__generate_content`
- `mcp__gemini-server__start_chat`
- `mcp__gemini-server__send_message`
- `mcp__gemini-server__list_files`
- `mcp__gemini-server__delete_file`

## What the Server Provides

### Tools (6 total)

1. **upload_file** - Upload files to Gemini Files API for processing
2. **generate_content** - Generate content using Gemini with optional file context
3. **start_chat** - Start a stateful chat session with Gemini
4. **send_message** - Send follow-up messages in an active chat session
5. **list_files** - List all uploaded files with metadata and expiration
6. **delete_file** - Delete files from Gemini Files API

### Resources (2 total)

1. **chat-history** - Retrieve metadata for active chat sessions
2. **file-info** - Retrieve metadata for uploaded files

### Prompts (3 total)

1. **analyze_media** - Template for analyzing audio, video, images, or documents
2. **document_qa** - Template for document question-answering (summary/detailed/extraction)
3. **multimodal_analysis** - Template for comparing or synthesizing multiple files

## Usage Examples

### Example 1: Simple Chat

```
User: I want to chat with Gemini
Claude Code: I'll start a chat session with Gemini
Tool Call: start_chat(session_id="my-session", initial_message="Hello")
Response: "Hello! How can I help you?"

Tool Call: send_message(session_id="my-session", message="Tell me a joke")
Response: "Why did the scarecrow win an award? Because he was outstanding in his field!"
```

### Example 2: File Upload and Analysis

```
User: Analyze this image using Gemini
Claude Code: I'll upload the image and analyze it
Tool Call: upload_file(file_uri="/path/to/image.jpg")
Response: file_uri = "https://generativelanguage.googleapis.com/..."

Tool Call: generate_content(prompt="What's in this image?", file_uris=[file_uri])
Response: "This image shows a cat sitting on a windowsill..."
```

## Troubleshooting

### Issue: Tools not appearing in Claude Code

**Cause:** Configuration not loaded or syntax error

**Solutions:**
1. Verify JSON syntax: `python -m json.tool < ~/.config/Claude/claude_desktop_config.json`
2. Check file permissions: `chmod 644 ~/.config/Claude/claude_desktop_config.json`
3. Completely quit and restart Claude Code (not just close window)
4. Check that paths are absolute (not relative)

### Issue: "google-genai SDK not available"

**Cause:** Python interpreter doesn't have the SDK in its path

**Solutions:**
1. Verify you're using the venv Python:
   ```bash
   /mnt/Shared/cs-projects/simply-mcp-py/venv/bin/python -c "import google.genai"
   ```
2. Check configuration uses correct Python path
3. Re-create venv if needed:
   ```bash
   cd /mnt/Shared/cs-projects/simply-mcp-py
   python3 -m venv venv
   source venv/bin/activate
   pip install -r demo/gemini/requirements.txt
   ```

### Issue: "GEMINI_API_KEY not set"

**Cause:** API key not provided to the server

**Solutions:**
1. Create `.env` file in `demo/gemini/` directory:
   ```bash
   echo "GEMINI_API_KEY=your-key-here" > demo/gemini/.env
   ```
2. Or add to MCP configuration:
   ```json
   "env": {
     "GEMINI_API_KEY": "your-key-here"
   }
   ```

### Issue: Server fails to start

**Cause:** Missing dependencies or configuration issue

**Solutions:**
1. Test server manually:
   ```bash
   /mnt/Shared/cs-projects/simply-mcp-py/venv/bin/python \
     /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py
   ```
2. Check for errors in the output
3. Verify venv is properly set up:
   ```bash
   source /mnt/Shared/cs-projects/simply-mcp-py/venv/bin/activate
   pip list | grep google-genai
   ```

## Architecture Overview

```
Claude Code
    ↓
MCP Configuration
    ↓
Spawn Server Process
    (venv Python)
    ↓
Gemini Server
    (stdio transport)
    ├─ 6 Tools
    ├─ 2 Resources
    └─ 3 Prompts
    ↓
Gemini API
```

## Key Design Decisions

1. **Venv Python:** Uses virtual environment to ensure all dependencies are available
2. **Module-level functions:** Refactored to eliminate MCP introspection issues
3. **Configuration-based:** API key and settings can be provided via environment
4. **Stateful sessions:** Chat sessions maintained on server-side with CHAT_SESSIONS registry
5. **File registry:** Uploaded files tracked with metadata and expiration (48 hours)

## Security Considerations

1. **API Key:** Store securely in `.env` file or environment variables (not in git)
2. **File uploads:** Automatically deleted after 48 hours by Gemini API
3. **Chat sessions:** Stored in-memory (cleared on server restart)
4. **Input validation:** All parameters validated before API calls

## Monitoring

The server logs to:
- **Console:** Initial startup messages
- **JSON logs:** Detailed operation logs in JSON format
- **File:** Can be configured to log to file for production

Check logs when debugging:
```bash
# Test and capture logs
timeout 10 /mnt/Shared/cs-projects/simply-mcp-py/venv/bin/python \
  /mnt/Shared/cs-projects/simply-mcp-py/demo/gemini/server.py 2>&1 | grep ERROR
```

## Next Steps

1. ✅ Complete all configuration steps above
2. ✅ Restart Claude Code completely
3. ✅ Test tools are available in Claude Code
4. ✅ Try using the tools (start_chat, send_message, upload_file)
5. ✅ Upload an image and analyze it
6. ✅ Test multi-turn conversations

## Support

For issues:
1. Check the troubleshooting section above
2. Verify configuration file syntax and paths
3. Test server manually using the venv Python
4. Check logs for detailed error messages
5. Ensure Claude Code is completely restarted after configuration changes

---

**Last Updated:** 2025-10-16
**Server Version:** 0.1.0
**Configuration:** MCP with venv Python
