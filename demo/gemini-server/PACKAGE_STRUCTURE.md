# Gemini MCP Server - Package Structure and Distribution

## Overview

The Gemini MCP Server is now properly packaged as a Python package following MCP best practices. This enables:
- **Distribution via PyPI**: Can be installed via `pip install gemini-mcp-server`
- **UVX Support**: Can be executed directly via `uvx gemini-mcp-server` without pre-installation
- **Dependency Management**: All dependencies are declared in `pyproject.toml` for automatic installation
- **Proper Entry Points**: CLI invocation through setuptools entry points

## Package Structure

```
demo/gemini-server/
├── pyproject.toml              # Package metadata and dependencies
├── README.md                   # User-facing documentation
├── PACKAGE_STRUCTURE.md        # This file
└── src/
    └── gemini_server/
        ├── __init__.py         # Package initialization and exports
        └── server.py           # Main server implementation (1600+ lines)
```

## Key Files

### `pyproject.toml`

Defines the package configuration:
- **Project metadata**: name, version, description, authors
- **Dependencies**:
  - `mcp>=0.1.0` - Model Context Protocol
  - `google-genai>=0.3.0` - Google Gemini API SDK
  - `pydantic>=2.0.0` - Data validation
  - `python-dotenv>=1.0.0` - Environment variable loading
  - `tomli>=2.0.0` - TOML parsing for Python <3.11

- **Entry point**: `gemini-server = "gemini_server.server:main"`
  - Maps the `gemini-server` CLI command to the `main()` function
  - Allows both `pip`-installed and `uvx` execution

### `src/gemini_server/__init__.py`

Package exports and metadata:
```python
__version__ = "0.1.0"
from gemini_server.server import create_gemini_server
```

Enables:
```python
from gemini_server import create_gemini_server
```

### `src/gemini_server/server.py`

The main server implementation (~1600 lines):

**Key Features:**
- **Lazy imports**: SDK only imported when actually needed, enabling execution in constrained environments
- **6 Tools**:
  - `upload_file` - Upload files to Gemini Files API
  - `generate_content` - Generate content with optional file context
  - `start_chat` - Start chat sessions
  - `send_message` - Continue chat sessions
  - `list_files` - List uploaded files
  - `delete_file` - Delete files

- **2 Resources**:
  - `chat-history://{session_id}` - Get session metadata
  - `file-info://{file_name}` - Get file metadata

- **3 Prompt Templates**:
  - `analyze_media` - Media analysis prompts
  - `document_qa` - Document Q&A prompts
  - `multimodal_analysis` - Multi-file analysis prompts

**Entry Point:**
```python
def main() -> None:
    """Entry point for CLI/uvx invocation"""
    asyncio.run(_async_main())
```

## How It Works

### Via pip (Traditional Installation)

```bash
# Install from PyPI (or from source)
pip install gemini-mcp-server

# Run the server
gemini-server
```

The package is installed with all dependencies in the Python environment's site-packages.

### Via uvx (UVX Execution)

```bash
# Execute without installation
uvx gemini-mcp-server
```

UVX:
1. Checks if package is cached locally
2. If not cached, downloads from PyPI with all dependencies
3. Executes the `gemini-server` entry point
4. Caches for future use

### Via Claude Code (Recommended)

Update `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gemini-server": {
      "command": "uvx",
      "args": ["gemini-mcp-server"],
      "env": {
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Benefits:
- ✅ No pre-installation required
- ✅ Dependencies automatically managed by uvx
- ✅ Always uses latest version from PyPI
- ✅ Isolated execution environment
- ✅ Automatic caching for performance

## Dependency Management

### Build Dependencies (`pyproject.toml` `[build-system]`)
- `hatchling` - Modern Python build backend

### Runtime Dependencies (`pyproject.toml` `[project] dependencies`)
These are installed when the package is installed:

```toml
dependencies = [
    "mcp>=0.1.0",
    "google-genai>=0.3.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "tomli>=2.0.0; python_version < '3.11'",
]
```

### Lazy Loading
The server uses lazy imports to handle optional dependencies gracefully:

```python
def _ensure_genai_available() -> bool:
    """Import SDK only when needed"""
    if genai is not None:
        return GENAI_AVAILABLE

    try:
        from google import genai as _genai_module
        genai = _genai_module
        GENAI_AVAILABLE = True
        return True
    except ImportError:
        GENAI_AVAILABLE = False
        return False
```

This enables the server to:
- Work in environments where google-genai isn't pre-installed (when using uvx)
- Provide meaningful error messages if dependencies are missing
- Handle import failures gracefully

## Distribution

### To PyPI

```bash
# Build the package
python -m build

# Upload to PyPI
python -m twine upload dist/*
```

### Local Development

```bash
# Install in editable mode
pip install -e .

# Run the server
gemini-server
```

### From Source

```bash
# Clone and install from specific version
pip install git+https://github.com/Clockwork-Innovations/simply-mcp-py.git#subdirectory=demo/gemini-server@v0.1.0
```

## Configuration Precedence

1. **Environment variables** (highest priority)
   - `GEMINI_API_KEY`
   - `GEMINI_DEFAULT_MODEL`
   - etc.

2. **config.toml file** (if present)

3. **.env file** (if present)

4. **Hardcoded defaults** (lowest priority)

## Advantages of This Approach

### 1. Clean Separation
- Server code is isolated from the broader simply-mcp project
- Can be developed and distributed independently
- Reduces complexity for users who only want the Gemini server

### 2. Standard Python Packaging
- Follows Python Packaging Guide (PEP 517, PEP 518)
- Compatible with PyPI, pip, uv, poetry, and other tools
- Works with all Python package managers

### 3. UVX Compatibility
- Users don't need to pre-install Python packages
- Dependencies are automatically managed
- Server always runs with correct versions
- Reduces "dependency hell" issues

### 4. Easy Installation for Users
```bash
# Option 1: Traditional pip
pip install gemini-mcp-server

# Option 2: UVX (no installation needed)
uvx gemini-mcp-server

# Option 3: Via Claude Code (recommended)
# Add to claude_desktop_config.json
```

### 5. Production Ready
- Proper dependency versioning
- Clear entry points
- Semantic versioning
- Compatible with CI/CD pipelines

## Testing the Package

### Local Testing

```bash
# Install in editable mode
pip install -e .

# Run the server
export GEMINI_API_KEY="your-key"
gemini-server
```

### UVX Testing

```bash
# Build the package (optional)
python -m build

# Test with uvx
uvx --from /path/to/dist/gemini-mcp-server-0.1.0-py3-none-any.whl gemini-server
```

## Future Improvements

1. **Release to PyPI**: Publish official releases
2. **Version Management**: Implement semantic versioning
3. **Changelog**: Document version changes
4. **CI/CD**: Automate testing and releases
5. **Documentation**: Build comprehensive user guides
6. **Testing**: Add comprehensive test suite

## Related Documentation

- [README.md](./README.md) - User guide and API reference
- [pyproject.toml](./pyproject.toml) - Package configuration
- [../](../) - simply-mcp project root
