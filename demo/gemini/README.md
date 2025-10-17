# Gemini MCP Server

A comprehensive Model Context Protocol (MCP) server that provides seamless integration with Google's Gemini API. This server enables AI applications to upload files, generate content, and conduct interactive conversations using Google's advanced language models through a standardized MCP interface.

The Gemini MCP Server bridges the gap between MCP-compatible AI clients and Google's Gemini API, offering robust file handling capabilities including support for audio, video, images, and documents. With built-in state management, chat session tracking, and intelligent file registry, this server provides a production-ready foundation for building AI-powered applications that leverage Gemini's multimodal capabilities.

## Features

- **File Upload and Management** - Upload files up to 2GB to Gemini Files API with automatic MIME type detection
- **Content Generation** - Generate text content with optional file context using any Gemini model
- **Interactive Chat Sessions** - Create and maintain stateful chat conversations with session management
- **Multimodal Support** - Process audio, video, images, and documents through Gemini's unified API
- **File Registry** - Track uploaded files with metadata including expiration times and URIs
- **Resource Endpoints** - Access chat history and file information through MCP resources
- **Prompt Templates** - Pre-built templates for media analysis, document Q&A, and multimodal workflows
- **Automatic Expiration Handling** - Files automatically expire after 48 hours per Gemini API limits
- **Structured Error Handling** - Consistent error responses with helpful hints for troubleshooting
- **Comprehensive Logging** - JSON-formatted logs with contextual metadata for debugging

## Requirements

### System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, or Windows
- **API Key**: Google AI Studio API key ([get one here](https://aistudio.google.com/app/apikey))

### Dependencies

Core dependencies (automatically installed):

- `simply-mcp` - MCP server framework
- `google-genai>=0.3.0` - Official Google Gemini SDK
- `python-dotenv>=1.0.0` - Environment variable management
- `mcp>=0.1.0` - Model Context Protocol SDK
- `pydantic>=2.0.0` - Data validation and settings
- `click>=8.0.0` - CLI framework
- `rich>=13.0.0` - Terminal formatting

## Installation

### Option 1: Install from Repository

```bash
# Clone the repository
git clone https://github.com/Clockwork-Innovations/simply-mcp-py.git
cd simply-mcp-py

# Install the package with development dependencies
pip install -e .

# Install Gemini-specific dependencies
pip install google-genai python-dotenv
```

### Option 2: Install as Standalone Package

```bash
# Install simply-mcp and dependencies
pip install simply-mcp google-genai python-dotenv
```

### Configure API Key

Create a `.env` file in the `demo/gemini/` directory or set environment variables:

```bash
# Option 1: Create .env file
cat > demo/gemini/.env << EOF
GEMINI_API_KEY=your-api-key-here
EOF

# Option 2: Export environment variable
export GEMINI_API_KEY="your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Quick Start

### Running the Server

```bash
# Method 1: Using simply-mcp dev command (recommended for development)
simply-mcp dev demo/gemini/server.py

# Method 2: Run directly with Python
python demo/gemini/server.py

# Method 3: Run as packaged .pyz file
simply-mcp build demo/gemini/server.py -o gemini-server.pyz
simply-mcp run gemini-server.pyz
```

### Example Usage

Here's a simple example of using the Gemini MCP server:

```python
# Using MCP client to interact with the server

# 1. Upload a file
response = await client.call_tool("upload_file", {
    "file_uri": "/path/to/document.pdf",
    "display_name": "Research Paper"
})
# Returns: {"success": true, "file_uri": "https://...", "file_id": "file_001"}

# 2. Generate content with file context
response = await client.call_tool("generate_content", {
    "prompt": "Summarize the key findings from this research paper",
    "file_uris": ["https://generativelanguage.googleapis.com/v1beta/files/..."],
    "model": "gemini-2.5-flash",
    "temperature": 0.7
})
# Returns: {"success": true, "text": "Summary...", "usage": {...}}

# 3. Start a chat session
response = await client.call_tool("start_chat", {
    "session_id": "research-discussion",
    "initial_message": "What are the main conclusions?",
    "model": "gemini-2.5-flash"
})
# Returns: {"success": true, "response": "Based on the research...", "message_count": 1}

# 4. Continue the conversation
response = await client.call_tool("send_message", {
    "session_id": "research-discussion",
    "message": "Can you elaborate on the methodology?"
})
# Returns: {"success": true, "response": "The methodology...", "message_count": 2}
```

## API Reference

### Tools

The server provides 6 tools for interacting with the Gemini API:

#### `upload_file`

Upload a local file to the Gemini Files API for use in prompts and content generation.

**Parameters:**
- `file_uri` (string, required) - Local file path to upload
- `display_name` (string, optional) - Human-readable name for the file (defaults to filename)

**Returns:**
```json
{
  "success": true,
  "file_id": "file_001",
  "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/...",
  "file_name": "files/abc123xyz",
  "display_name": "document.pdf",
  "size": 1048576,
  "mime_type": "application/pdf",
  "expires_at": "2025-10-18T12:00:00",
  "message": "File uploaded successfully: file_001"
}
```

**Supported File Types:**
- **Audio**: MP3, WAV, M4A, MP4 (audio)
- **Video**: MP4
- **Images**: PNG, JPG, JPEG, GIF, WebP, BMP
- **Documents**: PDF, TXT, HTML, CSS, JS, JSON, XML, CSV

**Limitations:**
- Maximum file size: 2GB per file
- Files expire after 48 hours
- MIME type detected automatically from file extension

---

#### `generate_content`

Generate content using a Gemini model with optional file context.

**Parameters:**
- `prompt` (string, required) - Text prompt for content generation
- `file_uris` (array of strings, optional) - List of Gemini file URIs to include as context
- `model` (string, optional) - Gemini model to use (default: "gemini-2.5-flash")
- `temperature` (float, optional) - Sampling temperature 0.0-2.0 (higher = more creative)
- `max_tokens` (integer, optional) - Maximum tokens to generate

**Returns:**
```json
{
  "success": true,
  "text": "Generated content...",
  "model": "gemini-2.5-flash",
  "usage": {
    "prompt_tokens": 150,
    "candidates_tokens": 300,
    "total_tokens": 450
  },
  "message": "Content generated successfully"
}
```

**Available Models:**
- `gemini-2.5-flash` - Fast, efficient model (default)
- `gemini-1.5-pro` - Most capable model
- `gemini-1.5-flash` - Balanced speed and capability

---

#### `start_chat`

Initialize a new chat session with Gemini and send the first message.

**Parameters:**
- `session_id` (string, required) - Unique identifier for this chat session
- `initial_message` (string, required) - First message to send in the conversation
- `file_uris` (array of strings, optional) - List of Gemini file URIs to include as context
- `model` (string, optional) - Gemini model to use (default: "gemini-2.5-flash")

**Returns:**
```json
{
  "success": true,
  "session_id": "chat-001",
  "response": "Hello! I'm ready to help...",
  "model": "gemini-2.5-flash",
  "message_count": 1,
  "message": "Chat session started: chat-001"
}
```

**Note:** Session IDs must be unique. Attempting to start a chat with an existing session ID will return an error.

---

#### `send_message`

Send a message to an existing chat session and receive a response.

**Parameters:**
- `session_id` (string, required) - ID of the existing chat session
- `message` (string, required) - Message text to send
- `file_uris` (array of strings, optional) - Additional file URIs to include with this message

**Returns:**
```json
{
  "success": true,
  "session_id": "chat-001",
  "response": "Based on your question...",
  "message_count": 3,
  "message": "Message sent to session: chat-001"
}
```

**Error Cases:**
- Session not found: Returns error with hint to use `start_chat` first
- Invalid session ID: Returns error with current active sessions

---

#### `list_files`

List all uploaded files currently stored in the Gemini Files API.

**Parameters:** None

**Returns:**
```json
{
  "success": true,
  "files": [
    {
      "file_id": "file_001",
      "file_name": "files/abc123",
      "display_name": "document.pdf",
      "size": 1048576,
      "mime_type": "application/pdf",
      "uploaded_at": "2025-10-16T10:00:00",
      "expires_at": "2025-10-18T10:00:00",
      "uri": "https://generativelanguage.googleapis.com/v1beta/files/..."
    }
  ],
  "count": 1,
  "expired_removed": 0,
  "message": "Found 1 uploaded files"
}
```

**Features:**
- Automatically filters out expired files
- Removes expired files from registry
- Returns metadata including expiration status
- Sorted by upload time (newest first)

---

#### `delete_file`

Delete a file from the Gemini Files API and remove it from the local registry.

**Parameters:**
- `file_name` (string, required) - Gemini file name (e.g., "files/abc123xyz")

**Returns:**
```json
{
  "success": true,
  "file_id": "file_001",
  "file_name": "files/abc123",
  "message": "File deleted successfully: files/abc123"
}
```

**Note:** Use `list_files` to get the correct `file_name` for deletion. File names follow the format `files/{id}`.

---

### Resources

The server exposes 2 MCP resources for accessing state information:

#### `chat-history://{session_id}`

Retrieve metadata and status for a chat session.

**URI Template:** `chat-history://research-chat`

**Returns:**
```json
{
  "success": true,
  "session_id": "research-chat",
  "model": "gemini-2.5-flash",
  "created_at": "2025-10-16T10:30:00",
  "message_count": 5,
  "status": "active",
  "message": "Chat session research-chat is active with 5 message(s)"
}
```

**Note:** The Gemini SDK doesn't provide built-in message history storage, so this resource returns session metadata rather than full message history.

---

#### `file-info://{file_name}`

Retrieve detailed metadata for an uploaded file.

**URI Template:** `file-info://files/abc123`

**Returns:**
```json
{
  "success": true,
  "file_id": "file_001",
  "file_name": "files/abc123",
  "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/...",
  "display_name": "document.pdf",
  "size": 1048576,
  "mime_type": "application/pdf",
  "uploaded_at": "2025-10-16T10:00:00",
  "expires_at": "2025-10-18T10:00:00",
  "status": "active"
}
```

**Status Values:**
- `active` - File is available and not expired
- `expired` - File has passed its expiration time (48 hours)

---

### Prompts

The server provides 3 prompt templates for common workflows:

#### `analyze_media`

Template for analyzing uploaded media files (audio, video, images, documents).

**Parameters:**
- `media_type` (string, required) - Type of media: "audio", "video", "image", or "document"

**Returns:** Formatted prompt text customized for the media type

**Example Usage:**
```python
# Get audio analysis prompt
prompt = await client.get_prompt("analyze_media", {"media_type": "audio"})
# Use with uploaded audio file
response = await client.call_tool("generate_content", {
    "prompt": prompt,
    "file_uris": ["https://...audio.mp3"]
})
```

**Templates Include:**
- **Audio**: Content overview, speaker identification, transcripts, quality assessment
- **Video**: Visual analysis, key moments, audio content, production quality
- **Image**: Subject matter, composition, colors, text extraction
- **Document**: Summary, key points, structure, important details

---

#### `document_qa`

Template for document question-answering workflows.

**Parameters:**
- `question_type` (string, required) - Type: "summary", "detailed", or "extraction"

**Returns:** Formatted prompt text for the question type

**Question Types:**
- **summary**: Concise overview with key points and action items
- **detailed**: Comprehensive analysis with background, strengths, and recommendations
- **extraction**: Structured data extraction (names, dates, numbers, decisions)

**Example Usage:**
```python
# Get detailed analysis prompt
prompt = await client.get_prompt("document_qa", {"question_type": "detailed"})
response = await client.call_tool("generate_content", {
    "prompt": prompt,
    "file_uris": ["https://...document.pdf"]
})
```

---

#### `multimodal_analysis`

Template for analyzing multiple files together.

**Parameters:**
- `analysis_type` (string, required) - Type: "compare", "synthesize", or "timeline"

**Returns:** Formatted prompt text for the analysis type

**Analysis Types:**
- **compare**: Side-by-side comparison of files with similarities and differences
- **synthesize**: Unified analysis combining information from all files
- **timeline**: Chronological sequencing of events across files

**Example Usage:**
```python
# Get synthesis prompt
prompt = await client.get_prompt("multimodal_analysis", {"analysis_type": "synthesize"})
response = await client.call_tool("generate_content", {
    "prompt": prompt,
    "file_uris": ["https://...file1.pdf", "https://...file2.pdf", "https://...file3.pdf"]
})
```

---

## Architecture

### State Management

The server maintains two primary state registries:

#### File Registry (`FILE_REGISTRY`)

Global dictionary tracking all uploaded files with the following metadata:

```python
FILE_REGISTRY = {
    "file_001": {
        "name": "files/abc123xyz",           # Gemini file name
        "uri": "https://...",                 # Gemini file URI
        "display_name": "document.pdf",       # Human-readable name
        "local_path": "/path/to/file.pdf",   # Original local path
        "size": 1048576,                      # File size in bytes
        "mime_type": "application/pdf",       # Detected MIME type
        "uploaded_at": "2025-10-16T10:00:00", # Upload timestamp
        "expires_at": "2025-10-18T10:00:00"   # Expiration timestamp
    }
}
```

**Features:**
- Automatic expiration tracking (48-hour lifetime)
- MIME type auto-detection from file extensions
- Size tracking for quota management
- Maps internal file IDs to Gemini file URIs

#### Chat Session Registry (`CHAT_SESSIONS`)

Global dictionary storing active chat sessions:

```python
@dataclass
class ChatSession:
    session_id: str          # Unique session identifier
    model: str              # Gemini model being used
    chat: Any               # genai.chats.Chat object
    created_at: datetime    # Session creation timestamp
    message_count: int      # Total messages in session

CHAT_SESSIONS = {
    "session_id": ChatSession(...)
}
```

**Features:**
- Type-safe dataclass structure
- Message count tracking
- Model configuration per session
- Session lifetime management

### Client Initialization

The Gemini client is lazily initialized on first use:

```python
def _get_gemini_client() -> Any:
    """Get or create the Gemini client instance."""
    global GEMINI_CLIENT

    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    GEMINI_CLIENT = genai.Client(api_key=api_key)
    return GEMINI_CLIENT
```

**Benefits:**
- Single client instance (singleton pattern)
- Graceful handling of missing API key
- Reduced initialization overhead

### Error Handling Pattern

All tools follow a consistent error handling pattern that never raises exceptions:

```python
@mcp.tool(name="example_tool")
def example_tool(param: str) -> dict[str, Any]:
    # Check SDK availability
    if not GENAI_AVAILABLE:
        return {
            "success": False,
            "error": "google-genai SDK not available",
            "install_hint": "pip install google-genai"
        }

    # Check API key
    client = _get_gemini_client()
    if not client:
        return {
            "success": False,
            "error": "GEMINI_API_KEY environment variable not set",
            "hint": "Set GEMINI_API_KEY before using this tool"
        }

    try:
        # Tool implementation
        result = perform_operation()
        return {"success": True, "data": result}
    except Exception as e:
        return {
            "success": False,
            "error": f"Operation failed: {str(e)}"
        }
```

**Features:**
- Structured error responses with hints
- No uncaught exceptions
- Consistent response format
- Helpful troubleshooting guidance

---

## Configuration

### Environment Variables

Configure the server using environment variables in a `.env` file:

```bash
# Required: Google AI Studio API Key
GEMINI_API_KEY=your-api-key-here

# Optional: Default model for content generation and chat
GEMINI_DEFAULT_MODEL=gemini-2.5-flash

# Optional: File expiration warning threshold (hours before expiration)
GEMINI_FILE_WARNING_HOURS=24
```

### Configuration File

Create a `.env` file in the `demo/gemini/` directory using the provided template:

```bash
# Copy example configuration
cp demo/gemini/.env.example demo/gemini/.env

# Edit with your API key
nano demo/gemini/.env
```

### Runtime Configuration

The server automatically loads configuration in this order:
1. Environment variables from `.env` file
2. System environment variables
3. Default values

---

## Troubleshooting

### Common Issues

#### API Key Not Found

**Error:**
```json
{
  "success": false,
  "error": "GEMINI_API_KEY environment variable not set",
  "hint": "Set GEMINI_API_KEY before using this tool"
}
```

**Solution:**
```bash
# Check if .env file exists
ls -la demo/gemini/.env

# Set environment variable
export GEMINI_API_KEY="your-api-key-here"

# Or create .env file
echo "GEMINI_API_KEY=your-api-key-here" > demo/gemini/.env
```

---

#### SDK Not Installed

**Error:**
```json
{
  "success": false,
  "error": "google-genai SDK not available",
  "install_hint": "pip install google-genai"
}
```

**Solution:**
```bash
# Install the SDK
pip install google-genai

# Verify installation
python -c "import google.genai; print('SDK installed')"
```

---

#### File Not Found

**Error:**
```json
{
  "success": false,
  "error": "File not found: /path/to/file.pdf"
}
```

**Solution:**
```bash
# Check file exists
ls -lh /path/to/file.pdf

# Use absolute path
realpath /path/to/file.pdf

# Verify permissions
chmod +r /path/to/file.pdf
```

---

#### File Size Limit Exceeded

**Error:**
```text
Upload failed: File size exceeds 2GB limit
```

**Solution:**
- Split large files into smaller chunks
- Compress files before uploading
- Use video/audio compression tools
- Consider using lower resolution media

**Maximum Limits:**
- File size: 2GB per file
- Total storage: Check your Gemini API quota

---

#### File Expired

**Error:**
```json
{
  "success": false,
  "error": "File not found in registry: files/abc123",
  "hint": "Use list_files to see available files"
}
```

**Solution:**
- Files expire after 48 hours
- Re-upload the file using `upload_file`
- Check expiration times with `list_files`
- Implement file refresh logic for long-running applications

```python
# Check file expiration
files = await client.call_tool("list_files")
for file in files["files"]:
    expires = datetime.fromisoformat(file["expires_at"])
    if expires < datetime.now() + timedelta(hours=24):
        print(f"File {file['display_name']} expires soon!")
```

---

#### Chat Session Not Found

**Error:**
```json
{
  "success": false,
  "error": "Chat session not found: session-123",
  "hint": "Use start_chat to create a new session first"
}
```

**Solution:**
```python
# Start a new chat session first
response = await client.call_tool("start_chat", {
    "session_id": "session-123",
    "initial_message": "Hello!"
})

# Then send additional messages
response = await client.call_tool("send_message", {
    "session_id": "session-123",
    "message": "Follow-up question"
})
```

---

#### Rate Limiting

**Error:**
```text
Generation failed: API rate limit exceeded
```

**Solution:**
- Implement exponential backoff retry logic
- Reduce request frequency
- Upgrade your Gemini API quota
- Use caching for repeated queries

```python
import time

max_retries = 3
for attempt in range(max_retries):
    response = await client.call_tool("generate_content", {...})
    if response["success"]:
        break
    time.sleep(2 ** attempt)  # Exponential backoff
```

---

#### Invalid Model Name

**Error:**
```text
Generation failed: Invalid model name
```

**Solution:**
- Use supported model names:
  - `gemini-2.5-flash`
  - `gemini-1.5-pro`
  - `gemini-1.5-flash`
- Check [Gemini documentation](https://ai.google.dev/models/gemini) for latest models

---

## Examples

### Example 1: Document Analysis Workflow

Complete workflow for analyzing a PDF document with follow-up questions:

```python
# Step 1: Upload the document
upload_response = await client.call_tool("upload_file", {
    "file_uri": "/Users/john/documents/research-paper.pdf",
    "display_name": "Machine Learning Research Paper"
})

if not upload_response["success"]:
    print(f"Upload failed: {upload_response['error']}")
    exit(1)

file_uri = upload_response["file_uri"]
print(f"Uploaded file: {upload_response['display_name']}")
print(f"Expires at: {upload_response['expires_at']}")

# Step 2: Get document summary using prompt template
summary_prompt = await client.get_prompt("document_qa", {
    "question_type": "summary"
})

summary_response = await client.call_tool("generate_content", {
    "prompt": summary_prompt,
    "file_uris": [file_uri],
    "model": "gemini-1.5-pro",
    "temperature": 0.3  # Lower temperature for factual summary
})

print("\n=== Document Summary ===")
print(summary_response["text"])

# Step 3: Extract key information
extraction_prompt = await client.get_prompt("document_qa", {
    "question_type": "extraction"
})

extraction_response = await client.call_tool("generate_content", {
    "prompt": extraction_prompt,
    "file_uris": [file_uri],
    "model": "gemini-1.5-pro"
})

print("\n=== Extracted Information ===")
print(extraction_response["text"])

# Step 4: Start interactive Q&A session
chat_response = await client.call_tool("start_chat", {
    "session_id": "document-qa-session",
    "initial_message": "What are the main limitations of the proposed approach?",
    "file_uris": [file_uri],
    "model": "gemini-1.5-pro"
})

print(f"\n=== Q&A Session (Message 1) ===")
print(chat_response["response"])

# Step 5: Follow-up questions
follow_up = await client.call_tool("send_message", {
    "session_id": "document-qa-session",
    "message": "Can you suggest improvements to the methodology?"
})

print(f"\n=== Q&A Session (Message 2) ===")
print(follow_up["response"])

# Step 6: Check session status
session_info = await client.get_resource("chat-history://document-qa-session")
print(f"\nTotal messages exchanged: {session_info['message_count']}")
```

---

### Example 2: Video Analysis and Transcription

Analyze a video file with timestamp extraction and key moment identification:

```python
# Step 1: Upload video file
upload_response = await client.call_tool("upload_file", {
    "file_uri": "/Users/john/videos/presentation.mp4",
    "display_name": "Product Demo Video"
})

video_uri = upload_response["file_uri"]
print(f"Video uploaded: {upload_response['size']} bytes")

# Step 2: Comprehensive video analysis
video_prompt = await client.get_prompt("analyze_media", {
    "media_type": "video"
})

analysis_response = await client.call_tool("generate_content", {
    "prompt": video_prompt,
    "file_uris": [video_uri],
    "model": "gemini-2.5-flash",
    "temperature": 0.5
})

print("\n=== Video Analysis ===")
print(analysis_response["text"])

# Step 3: Extract specific information with custom prompts
custom_prompts = [
    "List all key moments with timestamps in this format: [MM:SS] - Description",
    "Transcribe any on-screen text or captions shown in the video",
    "Identify and describe all speakers or people appearing in the video",
    "What is the primary message or call-to-action of this video?"
]

for i, prompt in enumerate(custom_prompts, 1):
    response = await client.call_tool("generate_content", {
        "prompt": prompt,
        "file_uris": [video_uri],
        "model": "gemini-2.5-flash"
    })
    print(f"\n=== Analysis {i}: {prompt[:50]}... ===")
    print(response["text"])
    print(f"Tokens used: {response['usage']['total_tokens']}")

# Step 4: Interactive discussion about the video
chat_response = await client.call_tool("start_chat", {
    "session_id": "video-discussion",
    "initial_message": "What improvements would you suggest for this presentation?",
    "file_uris": [video_uri]
})

print(f"\n=== Video Discussion ===")
print(chat_response["response"])
```

---

### Example 3: Multi-File Comparison and Synthesis

Compare multiple documents and synthesize insights:

```python
# Step 1: Upload multiple files
documents = [
    "/Users/john/reports/q1-report.pdf",
    "/Users/john/reports/q2-report.pdf",
    "/Users/john/reports/q3-report.pdf"
]

uploaded_files = []
for doc in documents:
    response = await client.call_tool("upload_file", {
        "file_uri": doc,
        "display_name": doc.split("/")[-1]
    })
    uploaded_files.append({
        "name": response["display_name"],
        "uri": response["file_uri"]
    })
    print(f"Uploaded: {response['display_name']}")

# Step 2: Compare the documents
compare_prompt = await client.get_prompt("multimodal_analysis", {
    "analysis_type": "compare"
})

file_uris = [f["uri"] for f in uploaded_files]

comparison_response = await client.call_tool("generate_content", {
    "prompt": compare_prompt,
    "file_uris": file_uris,
    "model": "gemini-1.5-pro",
    "temperature": 0.4,
    "max_tokens": 2000
})

print("\n=== Quarterly Reports Comparison ===")
print(comparison_response["text"])

# Step 3: Create synthesized analysis
synthesize_prompt = await client.get_prompt("multimodal_analysis", {
    "analysis_type": "synthesize"
})

synthesis_response = await client.call_tool("generate_content", {
    "prompt": synthesize_prompt + "\n\nFocus on trends, patterns, and year-over-year growth.",
    "file_uris": file_uris,
    "model": "gemini-1.5-pro",
    "temperature": 0.4,
    "max_tokens": 3000
})

print("\n=== Synthesized Insights ===")
print(synthesis_response["text"])

# Step 4: Create timeline of events
timeline_prompt = await client.get_prompt("multimodal_analysis", {
    "analysis_type": "timeline"
})

timeline_response = await client.call_tool("generate_content", {
    "prompt": timeline_prompt,
    "file_uris": file_uris,
    "model": "gemini-1.5-pro"
})

print("\n=== Timeline of Key Events ===")
print(timeline_response["text"])

# Step 5: Interactive analysis session
chat_response = await client.call_tool("start_chat", {
    "session_id": "quarterly-analysis",
    "initial_message": "What are the top 3 concerns based on all quarterly reports?",
    "file_uris": file_uris
})

print(f"\n=== Analysis Discussion ===")
print(chat_response["response"])

# Continue with follow-up questions
questions = [
    "What actionable recommendations would you make for Q4?",
    "Are there any red flags or warning signs across the quarters?",
    "Which quarter showed the strongest performance and why?"
]

for question in questions:
    response = await client.call_tool("send_message", {
        "session_id": "quarterly-analysis",
        "message": question
    })
    print(f"\nQ: {question}")
    print(f"A: {response['response']}")
```

---

### Example 4: Audio Transcription and Analysis

Process audio files for transcription and content analysis:

```python
# Step 1: Upload audio file
upload_response = await client.call_tool("upload_file", {
    "file_uri": "/Users/john/recordings/interview.mp3",
    "display_name": "Customer Interview - Jan 2025"
})

audio_uri = upload_response["file_uri"]
print(f"Audio uploaded: {upload_response['display_name']}")
print(f"File expires at: {upload_response['expires_at']}")

# Step 2: Full audio analysis
audio_prompt = await client.get_prompt("analyze_media", {
    "media_type": "audio"
})

analysis_response = await client.call_tool("generate_content", {
    "prompt": audio_prompt,
    "file_uris": [audio_uri],
    "model": "gemini-2.5-flash",
    "temperature": 0.3
})

print("\n=== Audio Analysis ===")
print(analysis_response["text"])

# Step 3: Extract transcript with timestamps
transcript_prompt = """
Please provide a detailed transcript of this audio file with timestamps.

Format:
[MM:SS] Speaker: What they said

Include:
- All spoken words
- Speaker identification (Speaker 1, Speaker 2, etc.)
- Timestamps every 30 seconds or at speaker changes
- Note any significant pauses or non-verbal sounds
"""

transcript_response = await client.call_tool("generate_content", {
    "prompt": transcript_prompt,
    "file_uris": [audio_uri],
    "model": "gemini-2.5-flash"
})

print("\n=== Transcript ===")
print(transcript_response["text"])

# Step 4: Extract insights and action items
insights_prompt = """
Based on this audio interview, provide:

1. Key Insights (3-5 main takeaways)
2. Action Items (specific tasks mentioned)
3. Decisions Made (any conclusions or agreements)
4. Follow-up Questions (things that need clarification)
5. Sentiment Analysis (overall tone and emotional content)

Format as a structured report.
"""

insights_response = await client.call_tool("generate_content", {
    "prompt": insights_prompt,
    "file_uris": [audio_uri],
    "model": "gemini-1.5-pro",
    "temperature": 0.4
})

print("\n=== Interview Insights ===")
print(insights_response["text"])

# Step 5: Check file metadata
file_info = await client.get_resource(f"file-info://{upload_response['file_name']}")
print(f"\n=== File Status ===")
print(f"Status: {file_info['status']}")
print(f"Size: {file_info['size']} bytes")
print(f"Expires: {file_info['expires_at']}")

# Step 6: Start Q&A session about the interview
chat_response = await client.call_tool("start_chat", {
    "session_id": "interview-analysis",
    "initial_message": "What were the customer's main pain points?",
    "file_uris": [audio_uri],
    "model": "gemini-1.5-pro"
})

print(f"\n=== Interview Q&A ===")
print(chat_response["response"])

# Follow-up questions
follow_ups = [
    "What solutions did they seem most interested in?",
    "Were there any objections or concerns raised?",
    "What should be our next steps with this customer?"
]

for question in follow_ups:
    response = await client.call_tool("send_message", {
        "session_id": "interview-analysis",
        "message": question
    })
    print(f"\nQ: {question}")
    print(f"A: {response['response'][:200]}...")  # Truncate for brevity
```

---

## Warnings and Limitations

### File Expiration

**IMPORTANT:** All files uploaded to Gemini Files API automatically expire after 48 hours.

- Plan workflows accordingly
- Re-upload files for long-running applications
- Monitor expiration times using `list_files`
- Implement automatic refresh logic if needed

```python
# Example: Check and refresh expiring files
async def refresh_expiring_files(client, warning_hours=24):
    files_response = await client.call_tool("list_files")
    now = datetime.now()

    for file in files_response["files"]:
        expires = datetime.fromisoformat(file["expires_at"])
        hours_remaining = (expires - now).total_seconds() / 3600

        if hours_remaining < warning_hours:
            print(f"WARNING: File '{file['display_name']}' expires in {hours_remaining:.1f} hours")
            # Re-upload logic here
```

### File Size Limits

- **Maximum file size**: 2GB per file
- **Total quota**: Varies by API tier
- Large files take longer to upload and process
- Consider compression for large media files

### API Rate Limits

- Rate limits depend on your Gemini API tier
- Implement retry logic with exponential backoff
- Cache responses when possible
- Monitor usage to avoid quota exhaustion

### Model Availability

- Model names may change over time
- Check [Gemini documentation](https://ai.google.dev/models/gemini) for latest models
- Some models may have regional restrictions
- Preview/experimental models may be deprecated

### Session Persistence

- Chat sessions exist only in server memory
- Sessions are lost on server restart
- Implement session persistence if needed
- No built-in message history export

---

## License

This project is part of the simply-mcp framework and is licensed under the MIT License.

Copyright (c) 2025 Clockwork Innovations

See the [LICENSE](../../LICENSE) file in the project root for full license text.

---

## Support and Resources

- **Documentation**: [https://simply-mcp-py.readthedocs.io](https://simply-mcp-py.readthedocs.io)
- **GitHub Repository**: [https://github.com/Clockwork-Innovations/simply-mcp-py](https://github.com/Clockwork-Innovations/simply-mcp-py)
- **Issues**: [GitHub Issues](https://github.com/Clockwork-Innovations/simply-mcp-py/issues)
- **Gemini API Docs**: [https://ai.google.dev/docs](https://ai.google.dev/docs)
- **Get API Key**: [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Contributing

Contributions are welcome! Please see the main project [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Clockwork-Innovations/simply-mcp-py.git
cd simply-mcp-py

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python demo/gemini/validate.py

# Run linting
ruff check demo/gemini/
black demo/gemini/

# Run type checking
mypy demo/gemini/
```
