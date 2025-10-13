"""Unit tests for error handling system."""

import pytest

from simply_mcp.core.errors import (
    AuthenticationError,
    AuthorizationError,
    ConfigFileNotFoundError,
    ConfigFormatError,
    ConfigValidationError,
    ConfigurationError,
    ConnectionError,
    HandlerError,
    HandlerExecutionError,
    HandlerNotFoundError,
    InvalidHandlerSignatureError,
    MessageError,
    RateLimitExceededError,
    RequiredFieldError,
    SchemaValidationError,
    SecurityError,
    SimplyMCPError,
    TransportError,
    TransportNotSupportedError,
    TypeValidationError,
    ValidationError,
)


class TestSimplyMCPError:
    """Tests for SimplyMCPError base class."""

    def test_basic_error_creation(self) -> None:
        """Test creating a basic error."""
        error = SimplyMCPError("Test error")
        assert error.message == "Test error"
        assert error.code == "SIMPLY_MCP_ERROR"
        assert error.context == {}

    def test_error_with_code(self) -> None:
        """Test error with custom code."""
        error = SimplyMCPError("Test error", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"

    def test_error_with_context(self) -> None:
        """Test error with context."""
        context = {"key": "value", "number": 42}
        error = SimplyMCPError("Test error", context=context)
        assert error.context == context

    def test_to_dict(self) -> None:
        """Test error serialization to dictionary."""
        error = SimplyMCPError("Test error", code="TEST_CODE", context={"foo": "bar"})
        result = error.to_dict()
        assert result["error"] == "SimplyMCPError"
        assert result["message"] == "Test error"
        assert result["code"] == "TEST_CODE"
        assert result["context"] == {"foo": "bar"}

    def test_str_representation(self) -> None:
        """Test string representation."""
        error = SimplyMCPError("Test error", code="TEST_CODE")
        assert str(error) == "[TEST_CODE] Test error"

    def test_repr_representation(self) -> None:
        """Test repr representation."""
        error = SimplyMCPError("Test error", code="TEST_CODE")
        assert "SimplyMCPError" in repr(error)
        assert "Test error" in repr(error)
        assert "TEST_CODE" in repr(error)

    def test_exception_inheritance(self) -> None:
        """Test that SimplyMCPError is an Exception."""
        error = SimplyMCPError("Test error")
        assert isinstance(error, Exception)

    def test_exception_raising(self) -> None:
        """Test raising and catching the error."""
        with pytest.raises(SimplyMCPError) as exc_info:
            raise SimplyMCPError("Test error", code="TEST")
        assert exc_info.value.message == "Test error"
        assert exc_info.value.code == "TEST"


class TestConfigurationErrors:
    """Tests for configuration-related errors."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        error = ConfigurationError("Config error")
        assert error.message == "Config error"
        assert error.code == "CONFIG_ERROR"
        assert isinstance(error, SimplyMCPError)

    def test_config_file_not_found(self) -> None:
        """Test ConfigFileNotFoundError."""
        error = ConfigFileNotFoundError("/path/to/config.toml")
        assert "config.toml" in error.message
        assert error.code == "CONFIG_NOT_FOUND"
        assert error.context["file_path"] == "/path/to/config.toml"
        assert isinstance(error, ConfigurationError)

    def test_config_validation_error(self) -> None:
        """Test ConfigValidationError."""
        error = ConfigValidationError("Invalid port", field="transport.port")
        assert "Invalid port" in error.message
        assert error.code == "CONFIG_VALIDATION_FAILED"
        assert error.context["field"] == "transport.port"

    def test_config_validation_error_without_field(self) -> None:
        """Test ConfigValidationError without field."""
        error = ConfigValidationError("Invalid config")
        assert error.message == "Invalid config"
        assert "field" not in error.context

    def test_config_format_error(self) -> None:
        """Test ConfigFormatError."""
        error = ConfigFormatError(".yaml")
        assert ".yaml" in error.message
        assert error.code == "CONFIG_FORMAT_UNSUPPORTED"
        assert error.context["file_format"] == ".yaml"
        assert ".toml" in str(error.context["supported_formats"])

    def test_config_format_error_custom_formats(self) -> None:
        """Test ConfigFormatError with custom supported formats."""
        error = ConfigFormatError(".ini", supported_formats=[".toml", ".json", ".yaml"])
        assert error.context["supported_formats"] == [".toml", ".json", ".yaml"]


class TestTransportErrors:
    """Tests for transport-related errors."""

    def test_transport_error(self) -> None:
        """Test TransportError."""
        error = TransportError("Transport failed")
        assert error.message == "Transport failed"
        assert error.code == "TRANSPORT_ERROR"
        assert isinstance(error, SimplyMCPError)

    def test_connection_error(self) -> None:
        """Test ConnectionError."""
        error = ConnectionError("Connection failed", host="localhost", port=8080)
        assert "Connection failed" in error.message
        assert error.code == "CONNECTION_FAILED"
        assert error.context["host"] == "localhost"
        assert error.context["port"] == 8080

    def test_connection_error_minimal(self) -> None:
        """Test ConnectionError without host/port."""
        error = ConnectionError("Connection failed")
        assert "host" not in error.context
        assert "port" not in error.context

    def test_transport_not_supported(self) -> None:
        """Test TransportNotSupportedError."""
        error = TransportNotSupportedError("websocket")
        assert "websocket" in error.message
        assert error.code == "TRANSPORT_NOT_SUPPORTED"
        assert error.context["transport_type"] == "websocket"
        assert "stdio" in str(error.context["supported_types"])

    def test_transport_not_supported_custom_types(self) -> None:
        """Test TransportNotSupportedError with custom supported types."""
        error = TransportNotSupportedError(
            "grpc", supported_types=["stdio", "http", "sse", "websocket"]
        )
        assert error.context["supported_types"] == ["stdio", "http", "sse", "websocket"]

    def test_message_error(self) -> None:
        """Test MessageError."""
        error = MessageError("Failed to decode message", message_type="request")
        assert "decode" in error.message
        assert error.code == "MESSAGE_ERROR"
        assert error.context["message_type"] == "request"

    def test_message_error_without_type(self) -> None:
        """Test MessageError without message type."""
        error = MessageError("Failed to encode")
        assert "message_type" not in error.context


class TestHandlerErrors:
    """Tests for handler-related errors."""

    def test_handler_error(self) -> None:
        """Test HandlerError."""
        error = HandlerError("Handler failed")
        assert error.message == "Handler failed"
        assert error.code == "HANDLER_ERROR"
        assert isinstance(error, SimplyMCPError)

    def test_handler_not_found(self) -> None:
        """Test HandlerNotFoundError."""
        error = HandlerNotFoundError("my_tool", handler_type="tool")
        assert "my_tool" in error.message
        assert "Tool" in error.message
        assert error.code == "HANDLER_NOT_FOUND"
        assert error.context["handler_name"] == "my_tool"
        assert error.context["handler_type"] == "tool"

    def test_handler_not_found_default_type(self) -> None:
        """Test HandlerNotFoundError with default type."""
        error = HandlerNotFoundError("my_handler")
        assert "handler" in error.message.lower()

    def test_handler_execution_error(self) -> None:
        """Test HandlerExecutionError."""
        original = ValueError("Invalid input")
        error = HandlerExecutionError("my_tool", original)
        assert "my_tool" in error.message
        assert "Invalid input" in error.message
        assert error.code == "HANDLER_EXECUTION_FAILED"
        assert error.context["handler_name"] == "my_tool"
        assert error.context["original_error"] == "Invalid input"
        assert error.context["error_type"] == "ValueError"
        assert error.original_error is original

    def test_invalid_handler_signature(self) -> None:
        """Test InvalidHandlerSignatureError."""
        error = InvalidHandlerSignatureError(
            "my_tool", expected_signature="(a: int, b: int)", actual_signature="(a: str)"
        )
        assert "my_tool" in error.message
        assert "a: int, b: int" in error.message
        assert "a: str" in error.message
        assert error.code == "INVALID_HANDLER_SIGNATURE"
        assert error.context["handler_name"] == "my_tool"
        assert error.context["expected_signature"] == "(a: int, b: int)"
        assert error.context["actual_signature"] == "(a: str)"


class TestValidationErrors:
    """Tests for validation-related errors."""

    def test_validation_error(self) -> None:
        """Test ValidationError."""
        error = ValidationError("Validation failed")
        assert error.message == "Validation failed"
        assert error.code == "VALIDATION_ERROR"
        assert isinstance(error, SimplyMCPError)

    def test_schema_validation_error(self) -> None:
        """Test SchemaValidationError."""
        errors = ["Field 'name' is required", "Field 'age' must be positive"]
        error = SchemaValidationError(
            "Schema validation failed",
            schema_path="$.properties.name",
            validation_errors=errors,
        )
        assert error.code == "SCHEMA_VALIDATION_FAILED"
        assert error.context["schema_path"] == "$.properties.name"
        assert error.context["validation_errors"] == errors

    def test_schema_validation_error_minimal(self) -> None:
        """Test SchemaValidationError with minimal args."""
        error = SchemaValidationError("Schema validation failed")
        assert "schema_path" not in error.context
        assert "validation_errors" not in error.context

    def test_type_validation_error(self) -> None:
        """Test TypeValidationError."""
        error = TypeValidationError("age", "int", "str")
        assert "age" in error.message
        assert "int" in error.message
        assert "str" in error.message
        assert error.code == "TYPE_VALIDATION_FAILED"
        assert error.context["field_name"] == "age"
        assert error.context["expected_type"] == "int"
        assert error.context["actual_type"] == "str"

    def test_required_field_error(self) -> None:
        """Test RequiredFieldError."""
        error = RequiredFieldError("username")
        assert "username" in error.message
        assert error.code == "REQUIRED_FIELD_MISSING"
        assert error.context["field_name"] == "username"


class TestSecurityErrors:
    """Tests for security-related errors."""

    def test_security_error(self) -> None:
        """Test SecurityError."""
        error = SecurityError("Security violation")
        assert error.message == "Security violation"
        assert error.code == "SECURITY_ERROR"
        assert isinstance(error, SimplyMCPError)

    def test_authentication_error(self) -> None:
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid credentials", auth_type="api_key")
        assert "Invalid credentials" in error.message
        assert error.code == "AUTHENTICATION_FAILED"
        assert error.context["auth_type"] == "api_key"

    def test_authentication_error_default_message(self) -> None:
        """Test AuthenticationError with default message."""
        error = AuthenticationError()
        assert error.message == "Authentication failed"

    def test_authorization_error(self) -> None:
        """Test AuthorizationError."""
        error = AuthorizationError("Access denied", required_permission="admin")
        assert "Access denied" in error.message
        assert error.code == "AUTHORIZATION_FAILED"
        assert error.context["required_permission"] == "admin"

    def test_authorization_error_default_message(self) -> None:
        """Test AuthorizationError with default message."""
        error = AuthorizationError()
        assert error.message == "Insufficient permissions"

    def test_rate_limit_exceeded(self) -> None:
        """Test RateLimitExceededError."""
        error = RateLimitExceededError("Too many requests", limit=60, retry_after=30)
        assert "Too many requests" in error.message
        assert error.code == "RATE_LIMIT_EXCEEDED"
        assert error.context["limit"] == 60
        assert error.context["retry_after"] == 30

    def test_rate_limit_exceeded_default_message(self) -> None:
        """Test RateLimitExceededError with default message."""
        error = RateLimitExceededError()
        assert error.message == "Rate limit exceeded"


class TestErrorHierarchy:
    """Tests for exception hierarchy and inheritance."""

    def test_all_errors_inherit_from_base(self) -> None:
        """Test that all errors inherit from SimplyMCPError."""
        errors = [
            ConfigurationError("test"),
            ConfigFileNotFoundError("/test"),
            ConfigValidationError("test"),
            ConfigFormatError(".yaml"),
            TransportError("test"),
            ConnectionError("test"),
            TransportNotSupportedError("test"),
            MessageError("test"),
            HandlerError("test"),
            HandlerNotFoundError("test"),
            HandlerExecutionError("test", ValueError("test")),
            InvalidHandlerSignatureError("test", "a", "b"),
            ValidationError("test"),
            SchemaValidationError("test"),
            TypeValidationError("field", "int", "str"),
            RequiredFieldError("field"),
            SecurityError("test"),
            AuthenticationError("test"),
            AuthorizationError("test"),
            RateLimitExceededError("test"),
        ]

        for error in errors:
            assert isinstance(error, SimplyMCPError)
            assert isinstance(error, Exception)

    def test_config_errors_hierarchy(self) -> None:
        """Test configuration errors hierarchy."""
        errors = [
            ConfigFileNotFoundError("/test"),
            ConfigValidationError("test"),
            ConfigFormatError(".yaml"),
        ]

        for error in errors:
            assert isinstance(error, ConfigurationError)
            assert isinstance(error, SimplyMCPError)

    def test_transport_errors_hierarchy(self) -> None:
        """Test transport errors hierarchy."""
        errors = [
            ConnectionError("test"),
            TransportNotSupportedError("test"),
            MessageError("test"),
        ]

        for error in errors:
            assert isinstance(error, TransportError)
            assert isinstance(error, SimplyMCPError)

    def test_handler_errors_hierarchy(self) -> None:
        """Test handler errors hierarchy."""
        errors = [
            HandlerNotFoundError("test"),
            HandlerExecutionError("test", ValueError("test")),
            InvalidHandlerSignatureError("test", "a", "b"),
        ]

        for error in errors:
            assert isinstance(error, HandlerError)
            assert isinstance(error, SimplyMCPError)

    def test_validation_errors_hierarchy(self) -> None:
        """Test validation errors hierarchy."""
        errors = [
            SchemaValidationError("test"),
            TypeValidationError("field", "int", "str"),
            RequiredFieldError("field"),
        ]

        for error in errors:
            assert isinstance(error, ValidationError)
            assert isinstance(error, SimplyMCPError)

    def test_security_errors_hierarchy(self) -> None:
        """Test security errors hierarchy."""
        errors = [
            AuthenticationError("test"),
            AuthorizationError("test"),
            RateLimitExceededError("test"),
        ]

        for error in errors:
            assert isinstance(error, SecurityError)
            assert isinstance(error, SimplyMCPError)


class TestErrorCodes:
    """Tests for error code uniqueness and consistency."""

    def test_error_codes_are_unique(self) -> None:
        """Test that all error codes are unique."""
        errors = [
            SimplyMCPError("test", code="SIMPLY_MCP_ERROR"),
            ConfigurationError("test"),
            ConfigFileNotFoundError("/test"),
            ConfigValidationError("test"),
            ConfigFormatError(".yaml"),
            TransportError("test"),
            ConnectionError("test"),
            TransportNotSupportedError("test"),
            MessageError("test"),
            HandlerError("test"),
            HandlerNotFoundError("test"),
            HandlerExecutionError("test", ValueError("test")),
            InvalidHandlerSignatureError("test", "a", "b"),
            ValidationError("test"),
            SchemaValidationError("test"),
            TypeValidationError("field", "int", "str"),
            RequiredFieldError("field"),
            SecurityError("test"),
            AuthenticationError("test"),
            AuthorizationError("test"),
            RateLimitExceededError("test"),
        ]

        # Get all codes except base category codes (which can be shared)
        codes = [e.code for e in errors]

        # These are base category codes that can be shared
        base_codes = {
            "SIMPLY_MCP_ERROR",
            "CONFIG_ERROR",
            "TRANSPORT_ERROR",
            "HANDLER_ERROR",
            "VALIDATION_ERROR",
            "SECURITY_ERROR",
        }

        # Specific error codes should be unique
        specific_codes = [c for c in codes if c not in base_codes]
        assert len(specific_codes) == len(set(specific_codes))

    def test_error_codes_follow_convention(self) -> None:
        """Test that error codes follow naming convention."""
        errors = [
            ConfigFileNotFoundError("/test"),
            ConfigValidationError("test"),
            ConfigFormatError(".yaml"),
            ConnectionError("test"),
            TransportNotSupportedError("test"),
            MessageError("test"),
            HandlerNotFoundError("test"),
            HandlerExecutionError("test", ValueError("test")),
            InvalidHandlerSignatureError("test", "a", "b"),
            SchemaValidationError("test"),
            TypeValidationError("field", "int", "str"),
            RequiredFieldError("field"),
            AuthenticationError("test"),
            AuthorizationError("test"),
            RateLimitExceededError("test"),
        ]

        for error in errors:
            # Error codes should be uppercase with underscores
            assert error.code.isupper()
            assert " " not in error.code
            assert "-" not in error.code


class TestJSONSerialization:
    """Tests for JSON serialization of errors."""

    def test_simple_error_serialization(self) -> None:
        """Test serialization of simple error."""
        error = SimplyMCPError("Test error", code="TEST")
        data = error.to_dict()
        assert isinstance(data, dict)
        assert data["error"] == "SimplyMCPError"
        assert data["message"] == "Test error"
        assert data["code"] == "TEST"
        assert data["context"] == {}

    def test_error_with_context_serialization(self) -> None:
        """Test serialization of error with context."""
        error = ConfigFileNotFoundError("/path/to/config.toml")
        data = error.to_dict()
        assert data["error"] == "ConfigFileNotFoundError"
        assert "config.toml" in data["message"]
        assert data["code"] == "CONFIG_NOT_FOUND"
        assert data["context"]["file_path"] == "/path/to/config.toml"

    def test_complex_context_serialization(self) -> None:
        """Test serialization with complex context."""
        error = SchemaValidationError(
            "Validation failed",
            schema_path="$.properties.name",
            validation_errors=["error1", "error2"],
        )
        data = error.to_dict()
        assert data["context"]["schema_path"] == "$.properties.name"
        assert data["context"]["validation_errors"] == ["error1", "error2"]

    def test_serialization_preserves_error_type(self) -> None:
        """Test that serialization preserves error type name."""
        errors = [
            ConfigFileNotFoundError("/test"),
            HandlerNotFoundError("test"),
            AuthenticationError("test"),
        ]

        for error in errors:
            data = error.to_dict()
            assert data["error"] == error.__class__.__name__


class TestErrorCatching:
    """Tests for catching errors at different hierarchy levels."""

    def test_catch_specific_error(self) -> None:
        """Test catching a specific error type."""
        with pytest.raises(ConfigFileNotFoundError):
            raise ConfigFileNotFoundError("/test")

    def test_catch_category_error(self) -> None:
        """Test catching errors by category."""
        with pytest.raises(ConfigurationError):
            raise ConfigFileNotFoundError("/test")

    def test_catch_base_error(self) -> None:
        """Test catching any Simply-MCP error."""
        with pytest.raises(SimplyMCPError):
            raise HandlerNotFoundError("test")

    def test_catch_as_exception(self) -> None:
        """Test catching as generic Exception."""
        with pytest.raises(Exception):
            raise ValidationError("test")

    def test_multiple_error_handling(self) -> None:
        """Test handling multiple error types."""
        errors = [
            ConfigFileNotFoundError("/test"),
            HandlerNotFoundError("test"),
            AuthenticationError("test"),
        ]

        for error in errors:
            with pytest.raises(SimplyMCPError):
                raise error


class TestErrorContextPreservation:
    """Tests for context preservation in error handling."""

    def test_context_preserved_through_raise(self) -> None:
        """Test that context is preserved when error is raised."""
        try:
            raise ConfigFileNotFoundError("/path/to/config.toml")
        except ConfigFileNotFoundError as e:
            assert e.context["file_path"] == "/path/to/config.toml"

    def test_original_error_preserved(self) -> None:
        """Test that original error is preserved in HandlerExecutionError."""
        original = ValueError("Original error")
        try:
            raise HandlerExecutionError("my_tool", original)
        except HandlerExecutionError as e:
            assert e.original_error is original
            assert e.context["error_type"] == "ValueError"

    def test_multiple_context_fields(self) -> None:
        """Test multiple context fields preservation."""
        error = ConnectionError("Failed", host="localhost", port=8080)
        assert error.context["host"] == "localhost"
        assert error.context["port"] == 8080
        assert len(error.context) == 2
