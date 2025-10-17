# Gemini MCP Server Bundle

A Model Context Protocol (MCP) server that integrates Google's Gemini API, enabling AI assistants to upload files, generate content, and manage interactive chat sessions.

This is a **bundle** meant to be run with `uvx simply-mcp run`.

## Quick Start

### Run the Bundle

```bash
uvx simply-mcp run /path/to/demo/gemini-server/
```

The command will:
1. Download `simply-mcp` via uvx (cached on first run)
2. Create a virtual environment
3. Install dependencies from `pyproject.toml`
4. Find and run the Gemini MCP server
5. Listen on stdio for MCP protocol messages

## Features

- **File Management**: Upload, list, and delete files in the Gemini Files API
- **Content Generation**: Generate content with optional file context using Gemini models
- **Chat Sessions**: Start and manage multi-turn chat conversations with Gemini
- **Media Analysis**: Get pre-built prompt templates for analyzing audio, video, images, and documents
- **Lazy Imports**: Efficient dependency loading to work in constrained environments
- **Automatic Dependency Installation**: Dependencies declared in `pyproject.toml` are automatically installed by `simply-mcp run`

## Prerequisites

1. **Install uv** (required for dependency management)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Get a Gemini API Key**
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Create a new API key

3. **Set Environment Variable**
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

## Running the Bundle

### Direct Execution

```bash
export GEMINI_API_KEY="your-api-key-here"
uvx simply-mcp run /path/to/demo/gemini-server/
```

### With Custom Virtual Environment

Reuse a virtual environment for faster subsequent runs:

```bash
uvx simply-mcp run /path/to/demo/gemini-server/ \
  --venv-path ~/.mcp-venvs/gemini-server/
```

## Configuration

The server supports configuration through environment variables (highest priority):

### Required

- `GEMINI_API_KEY`: Your Gemini API key

### Optional

- `GEMINI_DEFAULT_MODEL`: Model to use (default: gemini-2.5-flash)
- `GEMINI_FILE_WARNING_HOURS`: Warning threshold for file expiration (default: 24)
- `GEMINI_MAX_FILE_SIZE`: Maximum file size in bytes (default: 2GB)

Example:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_DEFAULT_MODEL="gemini-1.5-pro"
uvx simply-mcp run ./
```

## Available Tools

### 1. upload_file
Upload a file to Gemini Files API for use in prompts.

```python
{
  "file_uri": "/path/to/file",
  "display_name": "optional display name"
}
```

### 2. generate_content
Generate content using Gemini API with optional file context.

```python
{
  "prompt": "Your prompt here",
  "file_uris": ["gemini://files/..."],  # optional
  "model": "gemini-2.5-flash",
  "temperature": 0.7,  # optional
  "max_tokens": 2048  # optional
}
```

### 3. start_chat
Start a new chat session with Gemini.

```python
{
  "session_id": "unique-session-id",
  "initial_message": "Your first message",
  "file_uris": [],  # optional
  "model": "gemini-2.5-flash"
}
```

### 4. send_message
Send a message in an existing chat session.

```python
{
  "session_id": "session-id",
  "message": "Your message",
  "file_uris": []  # optional
}
```

### 5. list_files
List all uploaded files with metadata.

```python
{}
```

### 6. delete_file
Delete a file from Gemini Files API.

```python
{
  "file_name": "files/abc123"
}
```

## Available Resources

### chat-history://{session_id}
Get chat session history and metadata.

### file-info://{file_name}
Get file metadata and status information.

## Available Prompts

### analyze_media
Template for analyzing uploaded media files.
- Arguments: `media_type` (audio, video, image, document)

### document_qa
Template for document question-answering.
- Arguments: `question_type` (summary, detailed, extraction)

### multimodal_analysis
Template for analyzing multiple files together.
- Arguments: `analysis_type` (compare, synthesize, timeline)

## Usage with Claude Code

Edit `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "uvx",
      "args": [
        "simply-mcp",
        "run",
        "/path/to/demo/gemini-server/"
      ],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Then restart Claude Code. The server will:
1. Download `simply-mcp` (first run only, then cached)
2. Install Gemini server dependencies
3. Run the Gemini MCP server
4. Be available for use in Claude Code

No pre-installation required!

## Supported Models

- `gemini-2.5-flash` (default, fast and efficient)
- `gemini-1.5-pro` (more capable, slower)
- `gemini-1.5-flash` (balanced)
- `gemini-1.0-pro` (older model)

## File Handling

- Files are uploaded to Gemini's Files API and expire after 48 hours
- The server maintains a local registry of uploaded files
- Maximum file size: 2GB (configurable)
- Supported formats: audio, video, images, documents (PDF, text, etc.)

## Error Handling

The server provides detailed error messages for:
- Missing or invalid API key
- Missing dependencies (google-genai SDK)
- File not found errors
- API rate limits
- Invalid configurations

## Troubleshooting

### "google-genai SDK not available"
```bash
pip install google-genai
```

### "GEMINI_API_KEY environment variable not set"
```bash
export GEMINI_API_KEY="your-api-key"
```

### API Key Errors
- Verify the API key is valid at [Google AI Studio](https://aistudio.google.com/app/apikey)
- Ensure the API key is passed correctly as an environment variable
- Check that Gemini API is enabled in your Google project

## License

MIT

## Support

For issues and questions, visit the [GitHub repository](https://github.com/Clockwork-Innovations/simply-mcp-py/issues)
