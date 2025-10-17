# NPM/NPX Setup Guide for Gemini MCP Server

## Overview

This guide explains how to make the Gemini MCP server available via `npx simply-mcp run` command and prepare it for npm distribution.

## Current State

The `simply-mcp` package is already configured for npm:

**pyproject.toml** (Entry point):
```toml
[project.scripts]
simply-mcp = "simply_mcp.cli.main:cli"
```

This means when `simply-mcp` is installed via npm (via PyPI), users can run:
```bash
simply-mcp run /path/to/demo/gemini/server.py
```

## NPX Command Availability

### Once simply-mcp is Published to npm

After publishing `simply-mcp` to npm, users will be able to use:

```bash
# Run the Gemini server without installing
npx simply-mcp run /path/to/demo/gemini/server.py

# Or with full path
npx simply-mcp run ./demo/gemini/server.py

# With options
npx simply-mcp run ./demo/gemini/server.py \
  --transport http \
  --port 3000
```

## Publishing to npm (For Team/Public Distribution)

### Option 1: PyPI Publishing (Python)

Since `simply-mcp` is a Python package, it's published to PyPI (Python Package Index), not npm.

To make it available:

```bash
# 1. Build the package
python -m build

# 2. Publish to PyPI
python -m twine upload dist/*

# 3. Users install with pip/pipx
pip install simply-mcp
# or
pipx install simply-mcp
```

Then users can run:
```bash
simply-mcp run /path/to/server.py
```

### Option 2: NPM Publishing (Node.js Wrapper)

To make it available via npm (for convenience with nodejs developers):

```bash
# 1. Create a package.json at root (if not exists)
cat > package.json << 'EOF'
{
  "name": "simply-mcp",
  "version": "0.1.0",
  "description": "Modern Python framework for building MCP servers",
  "type": "module",
  "bin": {
    "simply-mcp": "dist/cli.js"
  },
  "engines": {
    "node": ">=16.0.0"
  },
  "dependencies": {
    "@anthropic-ai/sdk": "^0.20.0"
  }
}
EOF

# 2. Create a Node.js wrapper
mkdir -p dist
cat > dist/cli.js << 'EOF'
#!/usr/bin/env node
import { spawn } from 'child_process';
import { execSync } from 'child_process';

// Check if Python is available
try {
  execSync('python --version', { stdio: 'ignore' });
} catch {
  console.error('Error: Python 3.10+ is required. Please install Python.');
  process.exit(1);
}

// Get the Python script path (relative to node_modules)
const pythonScript = process.argv[2];
const args = process.argv.slice(3);

// Run the Python CLI
const proc = spawn('python', ['-m', 'simply_mcp.cli.main', ...args], {
  stdio: 'inherit',
  cwd: process.cwd()
});

proc.on('exit', (code) => {
  process.exit(code);
});
EOF

chmod +x dist/cli.js

# 3. Publish to npm
npm publish
```

Then users can:
```bash
npm install -g simply-mcp
# or
npx simply-mcp run /path/to/server.py
```

## Making Gemini Server Available via npm

### Current Setup

The Gemini server is available to users who have `simply-mcp` installed:

```bash
# Install simply-mcp
pip install simply-mcp
# or
pipx install simply-mcp

# Then run Gemini server
simply-mcp run /path/to/demo/gemini/server.py
```

### Distributing as .mcpb Bundle

For easier distribution (one-click installation), use the .mcpb format:

```bash
# 1. Create the bundle
mcpb pack demo/gemini -o gemini-server.mcpb

# 2. Publish to GitHub releases or npm
# Users can then:
# - Double-click the .mcpb file to install
# - Or: claude mcp install gemini-server.mcpb
```

## Installation Methods for Users

### Method 1: Via PyPI (Python)
```bash
pip install simply-mcp
simply-mcp run /path/to/demo/gemini/server.py
```

### Method 2: Via NPM (Node.js)
```bash
npm install -g simply-mcp
simply-mcp run /path/to/demo/gemini/server.py
```

### Method 3: Via .mcpb Bundle (One-Click)
```bash
# Download gemini-server.mcpb
# Double-click or: claude mcp install gemini-server.mcpb
```

### Method 4: Via Claude Code Configuration
```bash
# Manual configuration in ~/.config/Claude/claude_desktop_config.json
# See SETUP_GUIDE.md
```

## Publishing Workflow

### Step 1: Prepare Release

```bash
# 1. Update version in pyproject.toml
# 2. Update CHANGELOG
# 3. Create release notes

# 4. Build the package
python -m build

# 5. Verify build
twine check dist/*
```

### Step 2: Publish to PyPI

```bash
# Publish
python -m twine upload dist/*

# Verify
pip install --upgrade simply-mcp
simply-mcp --version
```

### Step 3: Publish to npm (Optional)

```bash
# Build Node.js wrapper
npm run build

# Publish
npm publish
```

### Step 4: Create Release Assets

```bash
# Build .pyz packages
simply-mcp build demo/gemini/server.py -o gemini-server.pyz

# Build .mcpb bundles
mcpb pack demo/gemini -o gemini-server.mcpb

# Create GitHub release with assets
gh release create v0.1.0 \
  gemini-server.pyz \
  gemini-server.mcpb \
  -t "Version 0.1.0" \
  -n "Release notes here"
```

## Current NPX Capability

Right now, you CAN use:

```bash
# If simply-mcp is installed globally
simply-mcp run /absolute/path/to/demo/gemini/server.py

# If installed locally
npx simple-mcp run /absolute/path/to/demo/gemini/server.py
```

## Future Enhancement: Global NPX Template

To create a truly one-command setup, you could create a template package:

```bash
# Users would run:
npx create-gemini-mcp

# This would:
# 1. Clone the Gemini server repository
# 2. Install dependencies
# 3. Set up Claude Code configuration
# 4. Prompt for API key
# 5. Start the server
```

## Documentation

- [NPX Registry](https://www.npmjs.com/) - npm package registry
- [PyPI](https://pypi.org/) - Python package registry
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [Twine](https://twine.readthedocs.io/) - PyPI publishing tool

## Summary

| Command | Status | Requires | Use When |
|---------|--------|----------|----------|
| `simply-mcp run` | ✅ Available | simply-mcp installed | Development, testing |
| `npx simply-mcp` | ✅ When published | npm package | Users without Python |
| `claude mcp add` | ✅ Available | Claude Code | Official integration |
| `npx create-gemini-mcp` | ❌ Not yet | Template package | Simplified setup |

---

**Current Recommendation:** Use `claude mcp add` for Claude Code integration, or `simply-mcp run` for direct server testing.
