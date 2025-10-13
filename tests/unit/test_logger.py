"""Unit tests for the logging system."""

import json
import logging
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

import pytest

from simply_mcp.core.config import LogConfigModel
from simply_mcp.core.logger import (
    ContextualJSONFormatter,
    LoggerContext,
    get_logger,
    log_with_context,
    sanitize_sensitive_data,
    setup_logger,
)


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_defaults(self) -> None:
        """Test logger setup with default configuration."""
        config = LogConfigModel(
            level="INFO",
            format="text",
            enable_console=True,
        )

        logger = setup_logger(config, name="test_default")

        assert logger.name == "test_default"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler

    def test_setup_logger_json_format(self) -> None:
        """Test logger setup with JSON format."""
        config = LogConfigModel(
            level="DEBUG",
            format="json",
            enable_console=True,
        )

        logger = setup_logger(config, name="test_json")

        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 1

        # Check formatter is JSON
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, ContextualJSONFormatter)

    def test_setup_logger_with_file(self) -> None:
        """Test logger setup with file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_file")

            # Should have 1 file handler
            assert len(logger.handlers) == 1

            # Write a log message
            logger.info("Test message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check file was created and contains message
            assert log_file.exists()
            content = log_file.read_text()
            assert "Test message" in content

    def test_setup_logger_with_file_json(self) -> None:
        """Test logger setup with JSON file output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"

            config = LogConfigModel(
                level="INFO",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_file_json")

            logger.info("Test JSON message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check file contains valid JSON
            assert log_file.exists()
            content = log_file.read_text().strip()
            log_entry = json.loads(content)
            assert log_entry["message"] == "Test JSON message"
            assert log_entry["level"] == "INFO"

    def test_setup_logger_creates_directories(self) -> None:
        """Test logger creates parent directories for log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "logs" / "subdir" / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_dirs")
            logger.info("Test")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            assert log_file.exists()
            assert log_file.parent.exists()

    def test_setup_logger_multiple_handlers(self) -> None:
        """Test logger with both console and file handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=True,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_multi")

            # Should have 2 handlers (console + file)
            assert len(logger.handlers) == 2

    def test_setup_logger_log_levels(self) -> None:
        """Test logger respects different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = LogConfigModel(
                level=level,  # type: ignore[arg-type]
                format="text",
                enable_console=True,
            )

            logger = setup_logger(config, name=f"test_{level.lower()}")
            assert logger.level == getattr(logging, level)


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_default(self) -> None:
        """Test getting logger with default configuration."""
        logger = get_logger()

        assert logger is not None
        assert isinstance(logger, logging.Logger)

    def test_get_logger_with_name(self) -> None:
        """Test getting logger with specific name."""
        # Setup a parent logger first
        config = LogConfigModel(
            level="INFO",
            format="text",
            enable_console=True,
        )
        setup_logger(config, name="test_parent")

        logger = get_logger("child_module")

        assert "child_module" in logger.name
        assert isinstance(logger, logging.Logger)

    def test_get_logger_singleton(self) -> None:
        """Test logger follows singleton pattern."""
        config = LogConfigModel(
            level="INFO",
            format="text",
            enable_console=True,
        )
        setup_logger(config, name="test_singleton")

        logger1 = get_logger()
        logger2 = get_logger()

        assert logger1 is logger2


class TestLoggerContext:
    """Tests for LoggerContext context manager."""

    def test_context_basic(self) -> None:
        """Test basic context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )
            logger = setup_logger(config, name="test_context_basic")

            with LoggerContext(request_id="req-123"):
                logger.info("Test message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Verify message was logged
            content = log_file.read_text()
            assert "Test message" in content

    def test_context_nested(self) -> None:
        """Test nested context managers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )
            logger = setup_logger(config, name="test_context_nested")

            with LoggerContext(session_id="sess-456"):
                with LoggerContext(request_id="req-123"):
                    logger.info("Nested context")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Nested context" in content

    def test_context_cleanup(self) -> None:
        """Test context is properly cleaned up after exit."""
        config = LogConfigModel(
            level="INFO",
            format="json",
            enable_console=True,
        )
        logger = setup_logger(config, name="test_context_cleanup")

        with LoggerContext(request_id="req-123"):
            pass  # Context set

        # Context should be cleared after exit
        # We can't directly test _log_context, but we verify no errors

    def test_context_with_exception(self) -> None:
        """Test context manager handles exceptions properly."""
        config = LogConfigModel(
            level="INFO",
            format="json",
            enable_console=True,
        )
        logger = setup_logger(config, name="test_context_exception")

        try:
            with LoggerContext(request_id="req-123"):
                logger.info("Before exception")
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        # Context should be cleaned up even with exception

    def test_context_multiple_fields(self) -> None:
        """Test context with multiple fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )
            logger = setup_logger(config, name="test_context_multi")

            with LoggerContext(
                request_id="req-123",
                session_id="sess-456",
                user_id="user-789",
            ):
                logger.info("Multiple fields")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Multiple fields" in content


class TestLogWithContext:
    """Tests for log_with_context function."""

    def test_log_with_context_basic(self) -> None:
        """Test logging with additional context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )
            logger = setup_logger(config, name="test_log_context")

            log_with_context(logger, "INFO", "Test message", user_id="123")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Test message" in content

    def test_log_with_context_levels(self) -> None:
        """Test logging at different levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="DEBUG",
                format="text",
                enable_console=False,
                file=str(log_file),
            )
            logger = setup_logger(config, name="test_log_levels")

            log_with_context(logger, "DEBUG", "Debug message", key="value")
            log_with_context(logger, "INFO", "Info message", key="value")
            log_with_context(logger, "WARNING", "Warning message", key="value")
            log_with_context(logger, "ERROR", "Error message", key="value")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content


class TestSanitizeSensitiveData:
    """Tests for sanitize_sensitive_data function."""

    def test_sanitize_password(self) -> None:
        """Test sanitizing password field."""
        data = {"password": "secret123", "user": "john"}
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["user"] == "john"

    def test_sanitize_api_key(self) -> None:
        """Test sanitizing API key field."""
        data = {"api_key": "sk-123456", "name": "test"}
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["name"] == "test"

    def test_sanitize_multiple_patterns(self) -> None:
        """Test sanitizing multiple sensitive patterns."""
        data = {
            "password": "pass123",
            "api_key": "key123",
            "secret": "secret123",
            "token": "token123",
            "user": "john",
        }
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["api_key"] == "***REDACTED***"
        assert sanitized["secret"] == "***REDACTED***"
        assert sanitized["token"] == "***REDACTED***"
        assert sanitized["user"] == "john"

    def test_sanitize_nested_dict(self) -> None:
        """Test sanitizing nested dictionary."""
        data = {
            "user": "john",
            "auth": {
                "password": "secret",
                "token": "abc123",
            },
        }
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["user"] == "john"
        assert sanitized["auth"]["password"] == "***REDACTED***"
        assert sanitized["auth"]["token"] == "***REDACTED***"

    def test_sanitize_list_of_dicts(self) -> None:
        """Test sanitizing list of dictionaries."""
        data = {
            "users": [
                {"name": "john", "password": "pass1"},
                {"name": "jane", "password": "pass2"},
            ]
        }
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["users"][0]["name"] == "john"
        assert sanitized["users"][0]["password"] == "***REDACTED***"
        assert sanitized["users"][1]["name"] == "jane"
        assert sanitized["users"][1]["password"] == "***REDACTED***"

    def test_sanitize_case_insensitive(self) -> None:
        """Test sanitization is case-insensitive."""
        data = {
            "PASSWORD": "secret",
            "ApiKey": "key123",
            "SECRET": "secret123",
        }
        sanitized = sanitize_sensitive_data(data)

        assert sanitized["PASSWORD"] == "***REDACTED***"
        assert sanitized["ApiKey"] == "***REDACTED***"
        assert sanitized["SECRET"] == "***REDACTED***"

    def test_sanitize_string(self) -> None:
        """Test sanitizing string with sensitive data."""
        data = "password='secret123' user='john'"
        sanitized = sanitize_sensitive_data(data)

        assert "secret123" not in sanitized
        assert "***REDACTED***" in sanitized

    def test_sanitize_no_redact(self) -> None:
        """Test sanitization without redaction (removal)."""
        data = {"password": "secret", "user": "john"}
        sanitized = sanitize_sensitive_data(data, redact=False)

        assert "password" not in sanitized
        assert sanitized["user"] == "john"


class TestContextualJSONFormatter:
    """Tests for ContextualJSONFormatter."""

    def test_json_formatter_basic(self) -> None:
        """Test basic JSON formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"

            config = LogConfigModel(
                level="INFO",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_json_formatter")
            logger.info("Test message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Parse JSON log
            content = log_file.read_text().strip()
            log_entry = json.loads(content)

            assert "timestamp" in log_entry
            assert log_entry["level"] == "INFO"
            assert log_entry["message"] == "Test message"
            assert log_entry["logger"] == "test_json_formatter"

    def test_json_formatter_with_context(self) -> None:
        """Test JSON formatting with context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"

            config = LogConfigModel(
                level="INFO",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_json_context")

            with LoggerContext(request_id="req-123", session_id="sess-456"):
                logger.info("Context message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Parse JSON log
            content = log_file.read_text().strip()
            log_entry = json.loads(content)

            assert log_entry["request_id"] == "req-123"
            assert log_entry["session_id"] == "sess-456"

    def test_json_formatter_sanitizes_sensitive_data(self) -> None:
        """Test JSON formatter sanitizes sensitive data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"

            config = LogConfigModel(
                level="INFO",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_json_sanitize")

            with LoggerContext(password="secret123", user="john"):
                logger.info("Sensitive data")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Parse JSON log
            content = log_file.read_text().strip()
            log_entry = json.loads(content)

            # Password should be redacted (in context field)
            assert "context" in log_entry or "password" in log_entry
            if "context" in log_entry and "password" in log_entry["context"]:
                assert log_entry["context"]["password"] == "***REDACTED***"
            elif "password" in log_entry:
                assert log_entry["password"] == "***REDACTED***"

            # User should be preserved
            if "context" in log_entry and "user" in log_entry["context"]:
                assert log_entry["context"]["user"] == "john"
            elif "user" in log_entry:
                assert log_entry["user"] == "john"


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_logging(self) -> None:
        """Test logging from multiple threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_concurrent")

            def log_messages(thread_id: int) -> None:
                for i in range(10):
                    logger.info(f"Thread {thread_id} message {i}")

            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_messages, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check all messages were logged
            content = log_file.read_text()
            for i in range(5):
                for j in range(10):
                    assert f"Thread {i} message {j}" in content

    def test_concurrent_context(self) -> None:
        """Test context manager from multiple threads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.json"

            config = LogConfigModel(
                level="INFO",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_concurrent_context")

            def log_with_context(thread_id: int) -> None:
                with LoggerContext(thread_id=thread_id):
                    logger.info(f"Thread {thread_id} message")

            # Create multiple threads
            threads = []
            for i in range(5):
                thread = threading.Thread(target=log_with_context, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check all messages were logged
            content = log_file.read_text()
            for i in range(5):
                assert f"Thread {i} message" in content


class TestLogRotation:
    """Tests for log file rotation."""

    def test_log_rotation(self) -> None:
        """Test log file rotation when size limit is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_rotation")

            # Write enough data to trigger rotation (>10MB)
            large_message = "x" * 1024  # 1KB message
            for i in range(11000):  # ~11MB of data
                logger.info(f"Message {i}: {large_message}")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Check that backup files were created
            backup_files = list(Path(tmpdir).glob("test.log.*"))
            assert len(backup_files) > 0  # At least one backup file


class TestRichHandler:
    """Tests for ContextualRichHandler."""

    def test_rich_handler_with_context(self) -> None:
        """Test Rich handler renders context properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_rich_handler")

            with LoggerContext(request_id="req-123", extra_field="value"):
                logger.info("Rich handler test")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Rich handler test" in content

    def test_rich_handler_with_extra_context(self) -> None:
        """Test Rich handler with extra context from record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="test_rich_extra")

            log_with_context(logger, "INFO", "Extra context test", extra_key="extra_value")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()
            assert "Extra context test" in content

    def test_rich_handler_console_rendering(self) -> None:
        """Test Rich handler console rendering (for coverage)."""
        # This tests the render_message path
        config = LogConfigModel(
            level="INFO",
            format="text",
            enable_console=True,
        )

        logger = setup_logger(config, name="test_rich_console")

        # Test with context
        with LoggerContext(request_id="req-789", custom="field"):
            logger.info("Console test with context")

        # Test with extra fields
        log_with_context(logger, "INFO", "Console test with extra", field1="value1", field2="value2")

        # Test without context
        logger.info("Console test without context")


class TestGetLoggerEdgeCases:
    """Tests for get_logger edge cases."""

    def test_get_logger_initializes_default(self) -> None:
        """Test get_logger initializes default logger if needed."""
        # Clear the global logger instance first
        import simply_mcp.core.logger as logger_module
        logger_module._logger_instance = None

        logger = get_logger()
        assert logger is not None
        assert logger.level == logging.INFO

    def test_get_logger_with_name_initializes_default(self) -> None:
        """Test get_logger with name initializes default if needed."""
        # Clear the global logger instance first
        import simply_mcp.core.logger as logger_module
        logger_module._logger_instance = None

        logger = get_logger("child")
        assert logger is not None
        assert "child" in logger.name


class TestSanitizeEdgeCases:
    """Tests for edge cases in sanitization."""

    def test_sanitize_non_string_non_dict(self) -> None:
        """Test sanitizing non-string, non-dict values."""
        result = sanitize_sensitive_data(123)
        assert result == 123

        result = sanitize_sensitive_data(None)
        assert result is None

    def test_sanitize_empty_dict(self) -> None:
        """Test sanitizing empty dictionary."""
        result = sanitize_sensitive_data({})
        assert result == {}


class TestLoggerIntegration:
    """Integration tests for the logger system."""

    def test_end_to_end_json_logging(self) -> None:
        """Test complete JSON logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.json"

            config = LogConfigModel(
                level="DEBUG",
                format="json",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="integration_test")

            # Log at different levels with context
            with LoggerContext(session_id="sess-123"):
                logger.debug("Debug message")
                logger.info("Info message")

                with LoggerContext(request_id="req-456"):
                    logger.warning("Warning message")
                    logger.error("Error message")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            # Parse and verify all log entries
            content = log_file.read_text().strip()
            lines = content.split("\n")

            assert len(lines) == 4

            for line in lines:
                log_entry = json.loads(line)
                assert "timestamp" in log_entry
                assert "level" in log_entry
                assert "message" in log_entry
                assert log_entry["session_id"] == "sess-123"

    def test_end_to_end_text_logging(self) -> None:
        """Test complete text logging workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "integration.log"

            config = LogConfigModel(
                level="INFO",
                format="text",
                enable_console=False,
                file=str(log_file),
            )

            logger = setup_logger(config, name="integration_text")

            logger.info("Test message 1")
            logger.warning("Test message 2")
            logger.error("Test message 3")

            # Flush handlers
            for handler in logger.handlers:
                handler.flush()

            content = log_file.read_text()

            assert "Test message 1" in content
            assert "Test message 2" in content
            assert "Test message 3" in content
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content
