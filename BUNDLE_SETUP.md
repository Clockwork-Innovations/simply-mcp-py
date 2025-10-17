# Simply-MCP Bundle Support with UVX

This document describes how to create and run MCP server bundles with automatic dependency management via `uvx`.

## Quick Start

### Command Format

```bash
uvx simply-mcp run {bundle_path}
```

### Example: Run Gemini Server Bundle

```bash
# Run the gemini-server bundle with all dependencies automatically installed
uvx simply-mcp run /path/to/demo/gemini-server/
```

## How It Works

When you run `uvx simply-mcp run {bundle}`:

1. **UVX Downloads simply-mcp**: First invocation downloads simply-mcp package with its dependencies
2. **Detection**: simply-mcp detects if the path is a bundle (directory with `pyproject.toml`)
3. **Discovery**: Finds the server entry point (looks for `src/{package}/server.py` or `server.py`)
4. **Virtual Environment**: Creates a temporary virtual environment using `uv venv`
5. **Installation**: Installs bundle dependencies using `uv pip install -e {bundle}`
6. **Execution**: Runs the server with all dependencies available
7. **Cleanup**: Venv can be reused or cleaned up between runs

## Bundle Structure

### Standard Bundle Layout

```
my-mcp-server/
├── pyproject.toml                  # Package metadata & dependencies
├── README.md                       # Documentation
└── src/
    └── my_mcp_server/
        ├── __init__.py
        └── server.py              # Main server implementation
```

### Simple Bundle Layout

```
my-mcp-server/
├── pyproject.toml                  # Package metadata & dependencies
├── README.md                       # Documentation
└── server.py                       # Main server implementation
```

## Creating a Bundle

### Step 1: Create Directory Structure

```bash
mkdir my-mcp-server
cd my-mcp-server
mkdir -p src/my_mcp_server
```

### Step 2: Create pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-mcp-server"
version = "0.1.0"
description = "My MCP Server Bundle"
authors = [{name = "Your Name", email = "you@example.com"}]
requires-python = ">=3.10"

# IMPORTANT: List all runtime dependencies here
dependencies = [
    "mcp>=0.1.0",
    "requests>=2.28.0",
    "pydantic>=2.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/my_mcp_server"]
```

### Step 3: Create Server Implementation

Create `src/my_mcp_server/server.py`:

```python
#!/usr/bin/env python3
"""My MCP Server"""

from simply_mcp import SimplyMCP

# Create server
mcp = SimplyMCP(
    name="my-server",
    version="0.1.0",
)

@mcp.tool()
def my_tool(message: str) -> str:
    """A simple tool"""
    return f"Received: {message}"

if __name__ == "__main__":
    mcp.run()
```

### Step 4: Create __init__.py

Create `src/my_mcp_server/__init__.py`:

```python
"""My MCP Server Package"""

__version__ = "0.1.0"
```

### Step 5: Test Bundle

```bash
cd my-mcp-server
uvx simply-mcp run .
```

## Using with Claude Code

Configure in `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "uvx",
      "args": [
        "simply-mcp",
        "run",
        "/path/to/my-mcp-server/"
      ],
      "env": {}
    }
  }
}
```

## Dependency Management

### Declaring Dependencies

In `pyproject.toml`, list all dependencies:

```toml
dependencies = [
    "mcp>=0.1.0",
    "requests>=2.28.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
]
```

### How Dependencies are Installed

1. **First Run**: UVX downloads simply-mcp and creates a virtual environment
2. **Bundle Dependencies**: UVX reads your `pyproject.toml` and installs dependencies
3. **Caching**: Dependencies are cached by uv for fast subsequent runs
4. **Isolated**: Each bundle has its own virtual environment or reuses shared cache

### Lazy Dependencies

For large dependencies that aren't always needed:

```python
# In your server code
def lazy_import():
    try:
        import heavy_dependency
        return heavy_dependency
    except ImportError:
        raise ImportError("Optional dependency not installed")

@mcp.tool()
def tool_using_heavy_dep():
    """Only imports heavy_dependency when called"""
    heavy_dep = lazy_import()
    return heavy_dep.process()
```

## Advanced: Custom Virtual Environment Path

By default, a temporary venv is created. To persist it:

```bash
uvx simply-mcp run /path/to/bundle/ --venv-path ~/.mcp-venvs/my-server/
```

This creates/reuses a venv at `~/.mcp-venvs/my-server/` for faster subsequent runs.

## Performance Tips

### 1. Use Persistent Venv

```bash
# First run (installs dependencies)
uvx simply-mcp run ./bundle/ --venv-path ./venv/

# Subsequent runs (reuses venv, instant startup)
uvx simply-mcp run ./bundle/ --venv-path ./venv/
```

### 2. Minimize Dependencies

Only list dependencies you actually use in `pyproject.toml`. Fewer dependencies = faster installation.

### 3. Use Binary Packages

When possible, use pre-built binary packages instead of source distributions. UVX/uv handles this automatically.

### 4. Pin Versions

For reproducible builds, pin versions in `pyproject.toml`:

```toml
dependencies = [
    "mcp==0.1.0",           # Exact version
    "requests>=2.28.0,<3",  # Range
]
```

## Examples

### Gemini Server Bundle

```bash
uvx simply-mcp run /path/to/demo/gemini-server/
```

The gemini-server bundle includes:
- `pyproject.toml` with dependencies: mcp, google-genai, pydantic, python-dotenv
- `src/gemini_server/server.py` with 6 tools and 3 resources
- Automatic installation of google-genai and other dependencies

### Local Development

For development, install locally:

```bash
cd my-mcp-server
pip install -e .
simply-mcp run .
```

## Troubleshooting

### "uv is not installed"

Install uv: https://docs.astral.sh/uv/getting-started/installation/

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### "No pyproject.toml found"

Ensure your bundle directory has a `pyproject.toml` file:

```bash
# Check if file exists
ls my-mcp-server/pyproject.toml

# Create if missing
touch my-mcp-server/pyproject.toml
```

### "No server.py found"

Server discovery looks for:
1. `src/{package_name}/server.py` (standard layout)
2. `server.py` (root layout)
3. `main.py` (alternate name)

Ensure one of these exists.

### Dependencies not installing

Check your `pyproject.toml` syntax:

```bash
# Validate TOML
pip install tomli
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"
```

### ImportError after installation

The dependencies were installed to the virtual environment but your code is running outside it. This happens when using `--venv-path`:

```bash
# Correct: Dependencies are in the venv
uvx simply-mcp run ./bundle/ --venv-path ./venv/

# The server runs inside the venv with all dependencies
```

## Workflow for Development

### Local Testing

```bash
cd my-mcp-server

# Install locally for development
pip install -e .

# Run with simply-mcp
simply-mcp run src/my_mcp_server/server.py

# Or run directly
python -m src.my_mcp_server.server
```

### Testing with UVX

```bash
# Test the bundle as users would use it
uvx simply-mcp run .
```

### Distribution

For sharing:
- Push to GitHub
- Document in README.md
- Share path or link: `uvx simply-mcp run https://github.com/user/my-mcp-server`

## Command Reference

### Run Bundle (Recommended)

```bash
uvx simply-mcp run /path/to/bundle/
```

**Features:**
- ✅ Auto-detect bundle
- ✅ Install dependencies
- ✅ Find server entry point
- ✅ Run with all dependencies

### Run Single File

```bash
uvx simply-mcp run server.py
```

### Run with Custom Port

```bash
uvx simply-mcp run /path/to/bundle/ --transport http --port 8080
```

### Run with Persistent Venv

```bash
uvx simply-mcp run /path/to/bundle/ --venv-path ~/.mcp-venvs/my-server/
```

## Architecture Diagram

```
User Command:
  uvx simply-mcp run ./my-bundle/
         │
         ├─→ UVX downloads simply-mcp (cached)
         │
         ├─→ simply-mcp run command executes
         │
         ├─→ Detects bundle (has pyproject.toml)
         │
         ├─→ Creates/reuses virtual environment
         │
         ├─→ Reads dependencies from pyproject.toml
         │
         ├─→ Installs dependencies via uv pip install
         │
         ├─→ Finds server entry point (server.py)
         │
         ├─→ Loads server module
         │
         └─→ Runs MCP server on stdio transport
```

## See Also

- [Simply-MCP Documentation](./README.md)
- [UVX Guide](https://docs.astral.sh/uv/guides/tools/)
- [pyproject.toml Specification](https://packaging.python.org/specifications/core-metadata/)
