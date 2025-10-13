"""Unit tests for configuration management."""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from simply_mcp.core.config import (
    AuthConfigModel,
    FeatureFlagsModel,
    LogConfigModel,
    RateLimitConfigModel,
    ServerMetadataModel,
    SimplyMCPConfig,
    TransportConfigModel,
    get_default_config,
    load_config,
    load_config_from_env,
    load_config_from_file,
    validate_config,
)


class TestServerMetadataModel:
    """Tests for ServerMetadataModel."""

    def test_valid_server_metadata(self) -> None:
        """Test valid server metadata creation."""
        metadata = ServerMetadataModel(name="test-server", version="1.0.0")
        assert metadata.name == "test-server"
        assert metadata.version == "1.0.0"
        assert metadata.description is None

    def test_server_metadata_with_optional_fields(self) -> None:
        """Test server metadata with optional fields."""
        metadata = ServerMetadataModel(
            name="test-server",
            version="1.0.0",
            description="A test server",
            author="Test Author",
            homepage="https://example.com",
        )
        assert metadata.description == "A test server"
        assert metadata.author == "Test Author"
        assert metadata.homepage == "https://example.com"

    def test_server_metadata_name_validation(self) -> None:
        """Test server metadata name validation."""
        with pytest.raises(ValidationError):
            ServerMetadataModel(name="", version="1.0.0")


class TestTransportConfigModel:
    """Tests for TransportConfigModel."""

    def test_default_transport_config(self) -> None:
        """Test default transport configuration."""
        config = TransportConfigModel()
        assert config.type == "stdio"
        assert config.host == "0.0.0.0"
        assert config.port == 3000

    def test_http_transport_config(self) -> None:
        """Test HTTP transport configuration."""
        config = TransportConfigModel(type="http", port=8080)
        assert config.type == "http"
        assert config.port == 8080

    def test_port_validation(self) -> None:
        """Test port number validation."""
        with pytest.raises(ValidationError):
            TransportConfigModel(port=0)
        with pytest.raises(ValidationError):
            TransportConfigModel(port=70000)


class TestRateLimitConfigModel:
    """Tests for RateLimitConfigModel."""

    def test_default_rate_limit(self) -> None:
        """Test default rate limit configuration."""
        config = RateLimitConfigModel()
        assert config.enabled is True
        assert config.requests_per_minute == 60
        assert config.burst_size == 10

    def test_custom_rate_limit(self) -> None:
        """Test custom rate limit configuration."""
        config = RateLimitConfigModel(
            enabled=False, requests_per_minute=120, burst_size=20
        )
        assert config.enabled is False
        assert config.requests_per_minute == 120
        assert config.burst_size == 20


class TestAuthConfigModel:
    """Tests for AuthConfigModel."""

    def test_default_auth_config(self) -> None:
        """Test default auth configuration."""
        config = AuthConfigModel()
        assert config.type == "none"
        assert config.enabled is False
        assert config.api_keys == []

    def test_api_key_auth(self) -> None:
        """Test API key authentication configuration."""
        config = AuthConfigModel(
            type="api_key", enabled=True, api_keys=["key1", "key2"]
        )
        assert config.type == "api_key"
        assert config.enabled is True
        assert len(config.api_keys) == 2

    def test_api_key_validation(self) -> None:
        """Test API key validation when enabled."""
        with pytest.raises(ValidationError):
            # Should fail: api_key type enabled but no keys provided
            AuthConfigModel(type="api_key", enabled=True, api_keys=[])


class TestLogConfigModel:
    """Tests for LogConfigModel."""

    def test_default_log_config(self) -> None:
        """Test default logging configuration."""
        config = LogConfigModel()
        assert config.level == "INFO"
        assert config.format == "json"
        assert config.enable_console is True
        assert config.file is None

    def test_custom_log_config(self) -> None:
        """Test custom logging configuration."""
        config = LogConfigModel(
            level="DEBUG",
            format="text",
            file="/var/log/server.log",
            enable_console=False,
        )
        assert config.level == "DEBUG"
        assert config.format == "text"
        assert config.file == "/var/log/server.log"
        assert config.enable_console is False


class TestFeatureFlagsModel:
    """Tests for FeatureFlagsModel."""

    def test_default_features(self) -> None:
        """Test default feature flags."""
        config = FeatureFlagsModel()
        assert config.enable_progress is True
        assert config.enable_binary_content is True
        assert config.max_request_size == 10485760  # 10MB

    def test_custom_features(self) -> None:
        """Test custom feature flags."""
        config = FeatureFlagsModel(
            enable_progress=False,
            enable_binary_content=False,
            max_request_size=5242880,  # 5MB
        )
        assert config.enable_progress is False
        assert config.enable_binary_content is False
        assert config.max_request_size == 5242880


class TestSimplyMCPConfig:
    """Tests for SimplyMCPConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = SimplyMCPConfig()
        assert config.server.name == "simply-mcp-server"
        assert config.server.version == "0.1.0"
        assert config.transport.type == "stdio"
        assert config.rate_limit.enabled is True
        assert config.auth.enabled is False
        assert config.logging.level == "INFO"
        assert config.features.enable_progress is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SimplyMCPConfig(
            server=ServerMetadataModel(name="custom-server", version="2.0.0"),
            transport=TransportConfigModel(type="http", port=8000),
        )
        assert config.server.name == "custom-server"
        assert config.server.version == "2.0.0"
        assert config.transport.type == "http"
        assert config.transport.port == 8000


class TestLoadConfigFromFile:
    """Tests for load_config_from_file."""

    def test_load_toml_config(self) -> None:
        """Test loading configuration from TOML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[server]
name = "toml-server"
version = "1.0.0"

[transport]
type = "http"
port = 8080

[logging]
level = "DEBUG"
""")
            f.flush()
            temp_path = f.name

        try:
            config = load_config_from_file(temp_path)
            assert config.server.name == "toml-server"
            assert config.server.version == "1.0.0"
            assert config.transport.type == "http"
            assert config.transport.port == 8080
            assert config.logging.level == "DEBUG"
        finally:
            Path(temp_path).unlink()

    def test_load_json_config(self) -> None:
        """Test loading configuration from JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "server": {"name": "json-server", "version": "2.0.0"},
                    "transport": {"type": "sse", "port": 9000},
                },
                f,
            )
            f.flush()
            temp_path = f.name

        try:
            config = load_config_from_file(temp_path)
            assert config.server.name == "json-server"
            assert config.server.version == "2.0.0"
            assert config.transport.type == "sse"
            assert config.transport.port == 9000
        finally:
            Path(temp_path).unlink()

    def test_load_nonexistent_file(self) -> None:
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config_from_file("/nonexistent/file.toml")

    def test_load_unsupported_format(self) -> None:
        """Test loading from unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            with pytest.raises(ValueError, match="Unsupported configuration file format"):
                load_config_from_file(f.name)


class TestLoadConfigFromEnv:
    """Tests for load_config_from_env."""

    def test_load_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("SIMPLY_MCP_SERVER__NAME", "env-server")
        monkeypatch.setenv("SIMPLY_MCP_SERVER__VERSION", "3.0.0")
        monkeypatch.setenv("SIMPLY_MCP_TRANSPORT__TYPE", "http")
        monkeypatch.setenv("SIMPLY_MCP_TRANSPORT__PORT", "7000")

        config = load_config_from_env()
        assert config.server.name == "env-server"
        assert config.server.version == "3.0.0"
        assert config.transport.type == "http"
        assert config.transport.port == 7000


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        config = load_config()
        assert config.server.name == "simply-mcp-server"
        assert config.transport.type == "stdio"

    def test_load_from_file_with_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading from file with environment override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[server]
name = "file-server"
version = "1.0.0"

[transport]
port = 3000
""")
            f.flush()
            temp_path = f.name

        try:
            monkeypatch.setenv("SIMPLY_MCP_TRANSPORT__PORT", "9999")

            config = load_config(temp_path, env_override=True)
            assert config.server.name == "file-server"
            # Environment should override file
            assert config.transport.port == 9999
        finally:
            Path(temp_path).unlink()

    def test_load_from_file_without_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading from file without environment override."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("""
[server]
name = "file-server"
version = "1.0.0"

[transport]
port = 3000
""")
            f.flush()
            temp_path = f.name

        try:
            monkeypatch.setenv("SIMPLY_MCP_TRANSPORT__PORT", "9999")

            config = load_config(temp_path, env_override=False)
            assert config.server.name == "file-server"
            # File should take precedence
            assert config.transport.port == 3000
        finally:
            Path(temp_path).unlink()


class TestValidateConfig:
    """Tests for validate_config."""

    def test_validate_valid_config_object(self) -> None:
        """Test validating a valid config object."""
        config = SimplyMCPConfig()
        assert validate_config(config) is True

    def test_validate_valid_config_dict(self) -> None:
        """Test validating a valid config dictionary."""
        config_dict = {
            "server": {"name": "test", "version": "1.0.0"},
            "transport": {"type": "stdio"},
        }
        assert validate_config(config_dict) is True

    def test_validate_invalid_config_dict(self) -> None:
        """Test validating an invalid config dictionary."""
        config_dict = {
            "server": {"name": "", "version": "1.0.0"},  # Empty name
        }
        with pytest.raises(ValidationError):
            validate_config(config_dict)

    def test_validate_wrong_type(self) -> None:
        """Test validating wrong type."""
        with pytest.raises(TypeError):
            validate_config("not a config")  # type: ignore[arg-type]


class TestGetDefaultConfig:
    """Tests for get_default_config."""

    def test_get_default_config(self) -> None:
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, SimplyMCPConfig)
        assert config.server.name == "simply-mcp-server"
        assert config.server.version == "0.1.0"
        assert config.transport.type == "stdio"
        assert config.rate_limit.enabled is True
        assert config.auth.enabled is False
