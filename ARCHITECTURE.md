# Simply-MCP Bundle Architecture

## Overview

This document explains how the new bundle-based architecture works with `uvx simply-mcp run`.

## The Problem We Solved

Previously:
- MCP servers needed to be run via Python files
- Dependencies had to be pre-installed in a virtual environment
- Users needed complex setup instructions
- Dependency management was manual

Now:
- MCP servers are **bundles** with `pyproject.toml`
- Dependencies are **automatically installed** by simply-mcp
- Single command: `uvx simply-mcp run {bundle}`
- Fully automated dependency management via `uv`

## Architecture

### Flow Diagram

```
User Command:
  uvx simply-mcp run ./gemini-server/
         │
         ├─→ UVX downloads simply-mcp (cached)
         │   Installs: mcp, click, rich, etc.
         │
         └─→ simply-mcp run command starts
              │
              ├─→ Detects bundle (pyproject.toml)
              │
              ├─→ Finds server entry point (src/.../server.py)
              │
              ├─→ Creates virtual environment (or reuses with --venv-path)
              │
              ├─→ Reads dependencies from pyproject.toml
              │   google-genai, pydantic, python-dotenv, etc.
              │
              ├─→ Installs dependencies via uv pip install
              │
              ├─→ Imports server module
              │   (now all dependencies are available)
              │
              ├─→ Detects MCP server (module-level `mcp` variable)
              │
              └─→ Runs server on stdio transport
                  Listens for MCP protocol messages
```

### Key Components

#### 1. UVX (uv execute)

- Downloads and caches Python packages
- Manages dependencies automatically
- Runs packages without system-wide installation
- Provides isolated execution environment

#### 2. Simply-MCP CLI

- Provides the `simply-mcp run {bundle}` command
- Detects bundle vs. single file
- Manages virtual environments
- Installs dependencies
- Runs the MCP server

#### 3. MCP Server Bundle

- Directory with `pyproject.toml`
- Declares all dependencies
- Contains server implementation
- Standalone and portable

#### 4. Bundle Dependencies

Automatically installed by simply-mcp:
```toml
dependencies = [
    "simply-mcp>=0.1.0",      # Framework
    "google-genai>=0.3.0",    # API SDK
    "pydantic>=2.0.0",        # Validation
    "python-dotenv>=1.0.0",   # Config
]
```

## Workflow

### Step 1: User Runs Command

```bash
export GEMINI_API_KEY="your-key"
uvx simply-mcp run /path/to/gemini-server/
```

### Step 2: UVX Bootstrap

- Checks if `simply-mcp` is cached locally
- If not, downloads from PyPI with dependencies
- Caches for future use (~50MB total)
- Invokes: `simply-mcp run /path/to/gemini-server/`

### Step 3: Simply-MCP Detection

```python
# In run.py
is_bundle = path.is_dir() and (path / "pyproject.toml").exists()

if is_bundle:
    # Bundle handling
    server_entry = find_bundle_server(path)  # Find server.py
    install_bundle_dependencies(path, venv)   # Install deps
    module = load_python_module(server_entry) # Load code
```

### Step 4: Dependency Installation

```bash
# Creates venv
uv venv /tmp/simply_mcp_venv_*/

# Installs bundle dependencies into venv
uv pip install -e /path/to/bundle/
# This reads pyproject.toml and installs everything
```

### Step 5: Server Execution

```python
# Server module is now imported with all dependencies available
from gemini_server.server import mcp

# Run the MCP server
await server.run_stdio()
```

## Benefits

### For Users

✅ **No Pre-Installation**
```bash
uvx simply-mcp run ./my-bundle/
# Works immediately, dependencies auto-installed
```

✅ **Automatic Dependency Management**
- Declared in `pyproject.toml`
- Always correct versions
- No "dependency hell"

✅ **Fast Subsequent Runs**
- UVX caches packages
- Venvs can be reused with `--venv-path`
- Typical startup: 1-2 seconds

✅ **Simple Configuration**
```json
{
  "command": "uvx",
  "args": ["simply-mcp", "run", "./bundle/"]
}
```

### For Developers

✅ **Standard Python Packaging**
- Uses `pyproject.toml` (PEP 517/518)
- Compatible with all package managers
- Follows Python best practices

✅ **Portable Bundles**
- Can be version controlled
- Share via GitHub/GitLab
- No installation overhead

✅ **Clear Dependency Declaration**
- All dependencies in one place
- Easy to update
- Compatible with dependency scanning tools

## Implementation Details

### Bundle Detection

```python
def is_bundle(path: Path) -> bool:
    return path.is_dir() and (path / "pyproject.toml").exists()
```

### Server Entry Point Discovery

Searches in order:
1. `src/{package_name}/server.py` (standard layout)
2. `server.py` (root layout)
3. `main.py` (alternate)

### Dependency Installation

```bash
# Create venv
uv venv {venv_path}

# Install bundle as editable with dependencies
uv pip install -e {bundle_path}
# This reads [project] dependencies from pyproject.toml
```

### Module Loading

```python
# Import server module
module = load_python_module(server_entry_point)

# Detect server (looks for module-level 'mcp' variable)
api_style, server = detect_api_style(module)

# All dependencies are now available
```

## File Structure

### Gemini Server Bundle Example

```
demo/gemini-server/
├── pyproject.toml              # ← Dependencies declared here
├── README.md
├── PACKAGE_STRUCTURE.md
├── UVX_SETUP.md
└── src/
    └── gemini_server/
        ├── __init__.py
        └── server.py           # ← Entry point found here
                                #   Contains: mcp = create_gemini_server()
```

### Key File: pyproject.toml

```toml
[project]
name = "gemini-mcp-server-bundle"
dependencies = [
    "simply-mcp>=0.1.0",
    "google-genai>=0.3.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]
```

### Key Code: server.py

```python
from simply_mcp import BuildMCPServer

mcp = BuildMCPServer(name="gemini-server")

# Tools, resources, prompts defined here

# Module-level mcp variable is what simply-mcp run finds
```

## Running with Claude Code

### Configuration

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
        "GEMINI_API_KEY": "your-key"
      }
    }
  }
}
```

### Startup Process

1. Claude Code reads config
2. Launches: `uvx simply-mcp run /path/to/gemini-server/`
3. UVX ensures simply-mcp is available (cached)
4. simply-mcp creates venv and installs dependencies
5. Server starts and connects via stdio
6. Claude Code sends/receives MCP messages

## Performance Characteristics

### First Run

- UVX downloads simply-mcp: ~10-20 seconds
- Create venv: ~2-3 seconds
- Install dependencies: ~10-30 seconds (depends on bundle)
- Start server: ~1-2 seconds
- **Total: 25-55 seconds**

### Subsequent Runs

- UVX uses cached simply-mcp: instant
- Reuse venv (with --venv-path): instant
- Start server: ~1-2 seconds
- **Total: 2-3 seconds**

### Memory

- UVX + simply-mcp: ~50MB
- Virtual environment overhead: ~10-20MB
- Bundle dependencies: varies (google-genai ~100MB)
- **Total: 160-200MB typical**

## Advanced Usage

### Custom Virtual Environment Path

```bash
# Persistent venv for faster subsequent runs
uvx simply-mcp run ./bundle/ \
  --venv-path ~/.mcp-venvs/my-server/
```

### HTTP Transport

```bash
uvx simply-mcp run ./bundle/ \
  --transport http \
  --port 8080
```

### SSE Transport

```bash
uvx simply-mcp run ./bundle/ \
  --transport sse \
  --port 8080
```

## Comparison

### Before (Traditional)

```bash
# User has to:
1. Clone repo
2. Create venv: python -m venv venv
3. Install deps: pip install -e .
4. Run: python -m my_package.server
5. Configure in Claude Code with path to Python

# Setup: 5-10 minutes
# Fragile: dependencies might not be installed
```

### After (Bundle)

```bash
# User just runs:
uvx simply-mcp run /path/to/bundle/

# Setup: 10 seconds (CLI setup in Claude Code)
# Robust: dependencies auto-installed
```

## Security Considerations

### Isolation

- Each bundle runs in its own virtual environment
- Dependencies can't interfere with system
- No global package registry writes

### Version Pinning

Recommended in `pyproject.toml`:

```toml
dependencies = [
    "google-genai==0.3.0",      # Exact version
    "pydantic>=2.0.0,<3.0.0",   # Version range
]
```

## Future Improvements

- [ ] Caching optimizations
- [ ] Multiple bundle types (Docker, WASM)
- [ ] Automatic dependency updates
- [ ] Bundle marketplace/registry
- [ ] Cross-platform testing

## Related Documentation

- [BUNDLE_SETUP.md](./BUNDLE_SETUP.md) - Complete bundle creation guide
- [demo/gemini-server/README.md](./demo/gemini-server/README.md) - Gemini server bundle
- [src/simply_mcp/cli/run.py](./src/simply_mcp/cli/run.py) - Implementation
