# Session Persistence & Database Integration

**Phase 1.3 - Complete Implementation**

This document covers the complete session persistence system with all three layers: Foundation, Feature, and Polish.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Foundation Layer: SQLite Storage](#foundation-layer-sqlite-storage)
4. [Feature Layer: Session Management](#feature-layer-session-management)
5. [Polish Layer: Multi-Backend Support](#polish-layer-multi-backend-support)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Migration Guide](#migration-guide)
9. [Deployment](#deployment)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The session persistence system provides comprehensive database integration for the Gemini MCP server, enabling:

- **Persistent chat sessions** across server restarts
- **Message history** storage and retrieval
- **File upload tracking** with expiration management
- **Multiple database backends** (SQLite, PostgreSQL, MongoDB)
- **Automatic session management** with lifecycle hooks
- **Cross-backend migration** tools
- **Export and backup** functionality

### Design Philosophy

The system follows a three-layer architecture:

1. **Foundation Layer**: Core storage abstractions and SQLite implementation
2. **Feature Layer**: Session lifecycle management, auto-save, migrations
3. **Polish Layer**: Enterprise backends (PostgreSQL, MongoDB) and migration tools

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│              (HTTP Server, MCP Tools, etc.)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                      Feature Layer                           │
│  ┌──────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ SessionManager   │  │  Migrations  │  │ Migration     │ │
│  │ - Auto-save      │  │ - Versioning │  │ Tools         │ │
│  │ - Lifecycle      │  │ - Schema mgmt│  │ - Export/Imp  │ │
│  │ - Export         │  └──────────────┘  │ - Backup      │ │
│  └──────────────────┘                    └───────────────┘ │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                    Foundation Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │            Abstract SessionStorage                    │   │
│  │  - save_session()    - load_session()                │   │
│  │  - list_sessions()   - delete_session()              │   │
│  │  - save_file_record() - load_file_record()           │   │
│  └─────────────┬────────────────────────┬───────────────┘   │
│                │                        │                    │
│  ┌─────────────┴──────┐   ┌────────────┴──────────┐        │
│  │ SQLiteStorage      │   │ PostgreSQLStorage      │        │
│  │ - File-based       │   │ - Connection pool      │        │
│  │ - No dependencies  │   │ - JSONB support        │        │
│  │ - WAL mode         │   │ - Full ACID            │        │
│  └────────────────────┘   └────────────────────────┘        │
│                            ┌────────────────────────┐        │
│                            │ MongoDBStorage         │        │
│                            │ - Document-based       │        │
│                            │ - Embedded messages    │        │
│                            │ - Horizontal scaling   │        │
│                            └────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Data Models

#### ChatSession
```python
@dataclass
class ChatSession:
    session_id: str
    model: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    messages: list[ChatMessage]
    metadata: dict[str, Any]
    status: str  # active, archived, expired
```

#### ChatMessage
```python
@dataclass
class ChatMessage:
    message_id: str
    session_id: str
    role: str  # user, assistant
    content: str
    created_at: datetime
    metadata: dict[str, Any]
```

#### UploadedFileRecord
```python
@dataclass
class UploadedFileRecord:
    file_id: str
    gemini_file_name: str
    gemini_file_uri: str
    display_name: str
    size: int
    mime_type: str
    uploaded_at: datetime
    expires_at: datetime
    metadata: dict[str, Any]
```

---

## Foundation Layer: SQLite Storage

The foundation layer provides the core storage abstraction and SQLite implementation.

### Features

- **File-based database** - No server required
- **Zero external dependencies** - Uses Python's built-in sqlite3
- **WAL mode** - Better concurrent access
- **Foreign key constraints** - Data integrity
- **JSON metadata** - Flexible schema extension
- **Indexes** - Optimized queries

### Database Schema

```sql
-- Sessions table
CREATE TABLE chat_sessions (
    session_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON
    status TEXT DEFAULT 'active'
);

-- Messages table
CREATE TABLE chat_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    metadata TEXT,  -- JSON
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        ON DELETE CASCADE
);

-- Files table
CREATE TABLE uploaded_files (
    file_id TEXT PRIMARY KEY,
    gemini_file_name TEXT NOT NULL,
    gemini_file_uri TEXT NOT NULL,
    display_name TEXT NOT NULL,
    size INTEGER,
    mime_type TEXT,
    uploaded_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    metadata TEXT  -- JSON
);

-- Indexes
CREATE INDEX idx_sessions_status ON chat_sessions(status);
CREATE INDEX idx_sessions_updated ON chat_sessions(updated_at DESC);
CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_files_expires ON uploaded_files(expires_at);
```

### Usage Example

```python
from storage import SQLiteSessionStorage, ChatSession

# Initialize storage
storage = SQLiteSessionStorage("sessions.db")
await storage.initialize()

# Create session
session = ChatSession.create("gemini-2.5-flash")
session.add_message("user", "Hello!")
session.add_message("assistant", "Hi there!")

# Save session
await storage.save_session(session)

# Load session
loaded = await storage.load_session(session.session_id)

# List sessions
active_sessions = await storage.list_sessions(status="active")

# Delete session
await storage.delete_session(session.session_id)

# Cleanup
await storage.close()
```

---

## Feature Layer: Session Management

The feature layer adds session lifecycle management on top of the storage backends.

### SessionManager Features

- **Auto-save** - Automatic persistence on message completion
- **Startup loading** - Recover sessions on server restart
- **Session expiry** - Automatic cleanup of old sessions
- **Export functionality** - JSON, text, markdown formats
- **Statistics** - Session and message counts
- **Archival** - Mark sessions as archived instead of deleting

### Usage Example

```python
from storage import SQLiteSessionStorage, SessionManager

# Create storage and manager
storage = SQLiteSessionStorage("sessions.db")
await storage.initialize()

manager = SessionManager(
    storage,
    auto_save_enabled=True,
    expiry_hours=72
)

# Load sessions on startup
stats = await manager.startup()
print(f"Loaded {stats['active_sessions']} sessions")

# Create new session
session = await manager.create_session("gemini-2.5-flash")

# Add message (auto-saved)
session.add_message("user", "Hello!")
await manager.auto_save(session)

# Export session
json_export = await manager.export_session(
    session.session_id,
    format="json"
)

# Export to file
await manager.export_to_file(
    session.session_id,
    Path("export.md"),
    format="markdown"
)

# Get statistics
stats = await manager.get_statistics()

# Cleanup expired sessions
expired = await manager.cleanup_expired_sessions()

# Shutdown (saves all cached sessions)
await manager.shutdown()
```

### Export Formats

#### JSON Format
```json
{
  "session_id": "abc-123",
  "model": "gemini-2.5-flash",
  "created_at": "2025-10-16T10:00:00",
  "message_count": 4,
  "messages": [
    {
      "role": "user",
      "content": "Hello!",
      "created_at": "2025-10-16T10:00:01"
    },
    {
      "role": "assistant",
      "content": "Hi there!",
      "created_at": "2025-10-16T10:00:02"
    }
  ]
}
```

#### Markdown Format
```markdown
# Chat Session: abc-123

**Model:** gemini-2.5-flash
**Created:** 2025-10-16 10:00:00
**Messages:** 4

---

### 👤 User (10:00:01)

Hello!

### 🤖 Assistant (10:00:02)

Hi there!
```

#### Text Format
```
Chat Session: abc-123
Model: gemini-2.5-flash
Created: 2025-10-16 10:00:00
Messages: 4

================================================================================

[10:00:01] USER:
Hello!

[10:00:02] ASSISTANT:
Hi there!
```

---

## Polish Layer: Multi-Backend Support

The polish layer adds enterprise database backends and migration tools.

### PostgreSQL Backend

Enterprise-grade SQL database with advanced features.

#### Features
- **Connection pooling** - asyncpg pool for performance
- **JSONB columns** - Efficient metadata storage and querying
- **Full-text search** - Built-in search capabilities
- **Replication** - Master-slave and streaming replication
- **ACID transactions** - Strong consistency guarantees

#### Setup

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb gemini_mcp

# Install Python driver
pip install asyncpg
```

#### Configuration

```yaml
storage:
  backend: "postgresql"
  connection_string: "postgresql://user:password@localhost:5432/gemini_mcp"
  min_pool_size: 5
  max_pool_size: 20
```

#### Usage

```python
from storage import PostgreSQLSessionStorage

storage = PostgreSQLSessionStorage(
    "postgresql://user:pass@localhost/gemini_mcp",
    min_pool_size=5,
    max_pool_size=20
)

await storage.initialize()

# Use same API as SQLite
await storage.save_session(session)
loaded = await storage.load_session(session_id)
```

### MongoDB Backend

Document-oriented NoSQL database for flexible schema.

#### Features
- **Document storage** - Natural fit for session data
- **Embedded messages** - Messages stored within session documents
- **Flexible schema** - No schema migrations needed
- **Horizontal scaling** - Built-in sharding support
- **Aggregation pipeline** - Complex queries and analytics

#### Setup

```bash
# Install MongoDB
sudo apt-get install mongodb

# Start service
sudo systemctl start mongodb

# Install Python driver
pip install motor
```

#### Configuration

```yaml
storage:
  backend: "mongodb"
  connection_string: "mongodb://localhost:27017"
  options:
    database_name: "gemini_mcp"
```

#### Usage

```python
from storage import MongoDBSessionStorage

storage = MongoDBSessionStorage(
    "mongodb://localhost:27017",
    database_name="gemini_mcp"
)

await storage.initialize()

# Use same API as SQLite
await storage.save_session(session)
loaded = await storage.load_session(session_id)
```

#### Document Structure

```json
{
  "_id": "session-abc-123",
  "session_id": "session-abc-123",
  "model": "gemini-2.5-flash",
  "created_at": ISODate("2025-10-16T10:00:00Z"),
  "updated_at": ISODate("2025-10-16T10:05:00Z"),
  "message_count": 2,
  "status": "active",
  "metadata": {},
  "messages": [
    {
      "message_id": "msg-1",
      "role": "user",
      "content": "Hello!",
      "created_at": ISODate("2025-10-16T10:00:01Z"),
      "metadata": {}
    },
    {
      "message_id": "msg-2",
      "role": "assistant",
      "content": "Hi there!",
      "created_at": ISODate("2025-10-16T10:00:02Z"),
      "metadata": {}
    }
  ]
}
```

---

## Configuration

### Configuration Sources

Configuration can be loaded from three sources (in priority order):

1. **Environment variables** (highest priority)
2. **Configuration files** (YAML or TOML)
3. **Programmatic defaults** (lowest priority)

### Environment Variables

```bash
# Backend selection
export STORAGE_BACKEND=sqlite              # sqlite, postgresql, mongodb
export STORAGE_CONNECTION_STRING=...       # For PostgreSQL/MongoDB
export STORAGE_DATABASE_PATH=sessions.db   # For SQLite

# Pool configuration
export STORAGE_MIN_POOL_SIZE=5
export STORAGE_MAX_POOL_SIZE=20

# Session management
export STORAGE_AUTO_SAVE=true
export STORAGE_EXPIRY_HOURS=72
```

### Configuration File (YAML)

```yaml
storage:
  backend: "sqlite"
  database_path: "./gemini_sessions.db"
  auto_save: true
  expiry_hours: 72
```

### Configuration File (TOML)

```toml
[storage]
backend = "sqlite"
database_path = "./gemini_sessions.db"
auto_save = true
expiry_hours = 72
```

### Programmatic Configuration

```python
from storage.config import StorageConfig, create_storage

# Create config
config = StorageConfig(
    backend="postgresql",
    connection_string="postgresql://localhost/gemini",
    min_pool_size=5,
    max_pool_size=20,
    auto_save=True,
    expiry_hours=72
)

# Create storage from config
storage = create_storage(config)
await storage.initialize()
```

---

## API Reference

### SessionStorage (Abstract Base Class)

All storage backends implement this interface.

#### Methods

```python
async def initialize() -> None
```
Initialize storage backend (create schema, connections, etc.)

```python
async def save_session(session: ChatSession) -> bool
```
Save a session with all messages. Returns True if successful.

```python
async def load_session(session_id: str) -> Optional[ChatSession]
```
Load a session by ID. Returns None if not found.

```python
async def list_sessions(
    status: Optional[str] = None,
    limit: int = 100
) -> list[ChatSession]
```
List sessions, optionally filtered by status.

```python
async def delete_session(session_id: str) -> bool
```
Delete a session. Returns True if deleted.

```python
async def save_file_record(file_record: UploadedFileRecord) -> bool
```
Save a file upload record.

```python
async def load_file_record(file_id: str) -> Optional[UploadedFileRecord]
```
Load a file record by ID.

```python
async def list_file_records(limit: int = 100) -> list[UploadedFileRecord]
```
List file records.

```python
async def delete_expired_files() -> int
```
Delete expired file records. Returns count deleted.

```python
async def close() -> None
```
Close connections and cleanup resources.

### SessionManager

High-level session management with auto-save and lifecycle hooks.

#### Methods

```python
async def startup() -> dict[str, Any]
```
Load all active sessions on startup. Returns statistics.

```python
async def auto_save(session: ChatSession) -> bool
```
Auto-save a session after update. Returns True if saved.

```python
async def get_session(session_id: str) -> Optional[ChatSession]
```
Get a session (from cache or storage).

```python
async def create_session(model: str) -> ChatSession
```
Create a new session.

```python
async def archive_session(session_id: str) -> bool
```
Archive a session (mark as archived but keep in storage).

```python
async def cleanup_expired_sessions() -> int
```
Clean up expired sessions. Returns count expired.

```python
async def export_session(
    session_id: str,
    format: Literal["json", "text", "markdown"] = "json"
) -> Optional[str]
```
Export session conversation history in specified format.

```python
async def export_to_file(
    session_id: str,
    output_path: Path,
    format: Literal["json", "text", "markdown"] = "json"
) -> bool
```
Export session to file.

```python
async def get_statistics() -> dict[str, Any]
```
Get session statistics.

```python
async def shutdown() -> None
```
Shutdown manager and save all cached sessions.

### Migration Tools

```python
async def migrate_storage(
    source: SessionStorage,
    destination: SessionStorage,
    verify: bool = True,
    progress_callback: Optional[callable] = None
) -> dict[str, Any]
```
Migrate all data from source to destination storage.

```python
async def verify_migration(
    source: SessionStorage,
    destination: SessionStorage
) -> dict[str, Any]
```
Verify that migration was successful.

```python
async def export_to_json(
    storage: SessionStorage,
    output_path: Path | str
) -> dict[str, Any]
```
Export all data to JSON file.

```python
async def import_from_json(
    storage: SessionStorage,
    input_path: Path | str
) -> dict[str, Any]
```
Import data from JSON file.

```python
async def backup_storage(
    storage: SessionStorage,
    backup_dir: Path | str
) -> dict[str, Any]
```
Create a timestamped backup.

---

## Migration Guide

### Migrating Between Backends

```python
from storage import (
    SQLiteSessionStorage,
    PostgreSQLSessionStorage,
    migrate_storage
)

# Create source and destination
sqlite_storage = SQLiteSessionStorage("sessions.db")
await sqlite_storage.initialize()

pg_storage = PostgreSQLSessionStorage(
    "postgresql://localhost/gemini"
)
await pg_storage.initialize()

# Migrate
result = await migrate_storage(
    source=sqlite_storage,
    destination=pg_storage,
    verify=True
)

print(f"Migrated {result['sessions_migrated']} sessions")
print(f"Migrated {result['files_migrated']} files")
print(f"Duration: {result['duration_seconds']:.2f}s")

# Cleanup
await sqlite_storage.close()
await pg_storage.close()
```

### Export/Import for Backup

```python
from storage import export_to_json, import_from_json

# Export
await export_to_json(storage, "backup.json")

# Import (to different backend)
await import_from_json(new_storage, "backup.json")
```

---

## Deployment

### Production Recommendations

#### SQLite
**Best for:** Development, small deployments, single-instance servers

**Configuration:**
```yaml
storage:
  backend: "sqlite"
  database_path: "/var/lib/gemini/sessions.db"
  auto_save: true
```

**Pros:**
- Zero setup
- No external dependencies
- Perfect for development

**Cons:**
- No horizontal scaling
- Limited concurrent writes

#### PostgreSQL
**Best for:** Production, high availability, multi-instance deployments

**Configuration:**
```yaml
storage:
  backend: "postgresql"
  connection_string: "postgresql://gemini:${DB_PASSWORD}@localhost:5432/gemini_mcp"
  min_pool_size: 10
  max_pool_size: 50
```

**Pros:**
- Full ACID compliance
- Excellent concurrent access
- Rich query capabilities
- Mature ecosystem

**Cons:**
- Requires PostgreSQL server
- More complex setup

#### MongoDB
**Best for:** Very large deployments, flexible schema requirements

**Configuration:**
```yaml
storage:
  backend: "mongodb"
  connection_string: "mongodb://gemini:${DB_PASSWORD}@localhost:27017"
  options:
    database_name: "gemini_mcp"
```

**Pros:**
- Horizontal scaling
- Flexible schema
- Built-in sharding

**Cons:**
- Requires MongoDB server
- Eventual consistency (in sharded setups)

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY demo/gemini/ ./gemini/
COPY src/ ./src/

# Set environment
ENV STORAGE_BACKEND=postgresql
ENV STORAGE_CONNECTION_STRING=postgresql://gemini:password@db:5432/gemini_mcp

CMD ["python", "-m", "gemini.http_server_with_persistence"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  gemini:
    build: .
    ports:
      - "8000:8000"
    environment:
      - STORAGE_BACKEND=postgresql
      - STORAGE_CONNECTION_STRING=postgresql://gemini:password@db:5432/gemini_mcp
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=gemini_mcp
      - POSTGRES_USER=gemini
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Troubleshooting

### Common Issues

#### Issue: "asyncpg not installed"

**Solution:**
```bash
pip install asyncpg
```

#### Issue: "motor not installed"

**Solution:**
```bash
pip install motor
```

#### Issue: SQLite database locked

**Solution:**
- Check if WAL mode is enabled
- Reduce concurrent write operations
- Consider migrating to PostgreSQL

#### Issue: Connection pool exhausted

**Solution:**
```python
config = StorageConfig(
    backend="postgresql",
    connection_string="...",
    max_pool_size=50  # Increase pool size
)
```

#### Issue: Session not persisting

**Solution:**
- Check auto_save is enabled
- Verify storage is initialized
- Check logs for errors
- Manually call `await manager.auto_save(session)`

### Performance Tuning

#### SQLite
```python
# Enable WAL mode
await connection.execute("PRAGMA journal_mode = WAL")

# Adjust cache size
await connection.execute("PRAGMA cache_size = -64000")  # 64MB

# Synchronous mode
await connection.execute("PRAGMA synchronous = NORMAL")
```

#### PostgreSQL
```sql
-- Connection pooling
max_pool_size = 50

-- Statement timeout
SET statement_timeout = '30s';

-- Work memory
SET work_mem = '64MB';
```

#### MongoDB
```python
# Connection pooling
client = motor_asyncio.AsyncIOMotorClient(
    connection_string,
    maxPoolSize=50,
    minPoolSize=10
)

# Read preference
db = client.get_database(
    "gemini_mcp",
    read_preference=ReadPreference.SECONDARY_PREFERRED
)
```

---

## Summary

Phase 1.3 provides comprehensive session persistence with:

### Foundation Layer
✅ Abstract SessionStorage base class
✅ SQLite implementation with connection pooling
✅ Full CRUD operations
✅ Schema management

### Feature Layer
✅ SessionManager with auto-save
✅ Startup loading and session recovery
✅ Session expiry management
✅ Export functionality (JSON, text, markdown)
✅ Migration system

### Polish Layer
✅ PostgreSQL backend
✅ MongoDB backend
✅ Storage configuration management
✅ Cross-backend migration tools
✅ Backup and restore

**Total Implementation:**
- ~2,400 lines of production code
- ~750 lines of tests
- Complete documentation
- All three layers functional

**Phase 1 Status:** ✅ **COMPLETE**
- Phase 1.1: HTTP Transport ✅
- Phase 1.2: Async File Upload ✅
- Phase 1.3: Session Persistence ✅
