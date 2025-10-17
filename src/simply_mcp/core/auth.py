"""Authentication system for HTTP transport.

This module provides API key management and bearer token validation
for securing HTTP MCP endpoints.

Features:
- API key storage and validation
- Bearer token extraction from Authorization header
- Per-key rate limit configuration
- Environment variable or dict-based key loading
- Proper error messages for validation failures

Example:
    >>> from simply_mcp.core.auth import ApiKeyManager
    >>>
    >>> # Load from environment variable (JSON format)
    >>> manager = ApiKeyManager()
    >>> manager.load_from_env("GEMINI_API_KEYS")
    >>>
    >>> # Validate a token
    >>> is_valid, key_info = manager.validate_token("sk_test_12345")
    >>> if is_valid:
    ...     print(f"Valid key: {key_info.name}")
"""

import json
import os
from dataclasses import dataclass

from simply_mcp.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ApiKey:
    """Represents an API key with metadata and rate limiting configuration.

    Attributes:
        key: The actual API key string (e.g., "sk_test_12345")
        name: Human-readable name for this key (e.g., "Production Key")
        rate_limit: Maximum requests allowed in the time window
        window_seconds: Time window for rate limiting (in seconds)
        enabled: Whether this key is currently active
    """

    key: str
    name: str
    rate_limit: int = 100
    window_seconds: int = 3600
    enabled: bool = True


class ApiKeyManager:
    """Manages API keys for authentication.

    This class handles loading, storing, and validating API keys.
    Keys can be loaded from environment variables (JSON format) or
    directly from a dictionary.

    Example:
        >>> manager = ApiKeyManager()
        >>>
        >>> # Load from JSON in environment
        >>> manager.load_from_env("API_KEYS")
        >>>
        >>> # Or load from dict
        >>> manager.load_from_dict({
        ...     "keys": [
        ...         {
        ...             "key": "sk_test_12345",
        ...             "name": "Test Key",
        ...             "rate_limit": 100,
        ...             "window_seconds": 3600
        ...         }
        ...     ]
        ... })
        >>>
        >>> # Validate a token
        >>> is_valid, key_info = manager.validate_token("sk_test_12345")
    """

    def __init__(self) -> None:
        """Initialize API key manager with empty key storage."""
        self._keys: dict[str, ApiKey] = {}
        logger.info("API key manager initialized")

    def load_from_env(self, env_var: str) -> int:
        """Load API keys from environment variable containing JSON.

        Expected JSON format:
        {
            "keys": [
                {
                    "key": "sk_test_12345",
                    "name": "Test Key",
                    "rate_limit": 100,
                    "window_seconds": 3600,
                    "enabled": true
                }
            ]
        }

        Args:
            env_var: Name of environment variable containing JSON

        Returns:
            Number of keys loaded

        Raises:
            ValueError: If JSON is invalid or keys are missing
        """
        logger.info(f"Loading API keys from environment variable: {env_var}")

        json_data = os.getenv(env_var)
        if not json_data:
            logger.warning(f"Environment variable {env_var} not set or empty")
            raise ValueError(f"Environment variable {env_var} not set")

        try:
            config = json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {env_var}: {e}")
            raise ValueError(f"Invalid JSON in environment variable {env_var}: {e}") from e

        return self.load_from_dict(config)

    def load_from_dict(self, config: dict) -> int:
        """Load API keys from a dictionary.

        Args:
            config: Dictionary containing 'keys' list with key configurations

        Returns:
            Number of keys loaded

        Raises:
            ValueError: If configuration is invalid
        """
        if "keys" not in config:
            logger.error("Configuration missing 'keys' field")
            raise ValueError("Configuration must contain 'keys' list")

        keys_data = config["keys"]
        if not isinstance(keys_data, list):
            logger.error("'keys' field must be a list")
            raise ValueError("'keys' field must be a list")

        loaded_count = 0
        for key_config in keys_data:
            try:
                api_key = ApiKey(
                    key=key_config["key"],
                    name=key_config.get("name", "Unnamed Key"),
                    rate_limit=key_config.get("rate_limit", 100),
                    window_seconds=key_config.get("window_seconds", 3600),
                    enabled=key_config.get("enabled", True),
                )

                self._keys[api_key.key] = api_key
                loaded_count += 1

                logger.debug(
                    f"Loaded API key: {api_key.name}",
                    extra={
                        "key_name": api_key.name,
                        "rate_limit": api_key.rate_limit,
                        "window_seconds": api_key.window_seconds,
                    }
                )

            except KeyError as e:
                logger.warning(f"Skipping invalid key configuration - missing field: {e}")
                continue

        logger.info(
            f"Loaded {loaded_count} API keys",
            extra={"keys_loaded": loaded_count}
        )

        return loaded_count

    def add_key(self, api_key: ApiKey) -> None:
        """Add a single API key to the manager.

        Args:
            api_key: ApiKey instance to add
        """
        self._keys[api_key.key] = api_key
        logger.debug(f"Added API key: {api_key.name}")

    def remove_key(self, key: str) -> bool:
        """Remove an API key from the manager.

        Args:
            key: API key string to remove

        Returns:
            True if key was removed, False if key didn't exist
        """
        if key in self._keys:
            key_name = self._keys[key].name
            del self._keys[key]
            logger.info(f"Removed API key: {key_name}")
            return True

        logger.warning("Attempted to remove non-existent key")
        return False

    def validate_token(self, token: str) -> tuple[bool, ApiKey | None]:
        """Validate an API token.

        Args:
            token: API key string to validate

        Returns:
            Tuple of (is_valid, key_info)
            - is_valid: True if token is valid and enabled
            - key_info: ApiKey object if valid, None otherwise
        """
        if not token:
            logger.debug("Empty token provided")
            return False, None

        # Look up key
        api_key = self._keys.get(token)

        if not api_key:
            logger.warning(
                "Invalid API key attempted",
                extra={"token_prefix": token[:8] + "..." if len(token) > 8 else token}
            )
            return False, None

        # Check if key is enabled
        if not api_key.enabled:
            logger.warning(
                f"Disabled API key attempted: {api_key.name}",
                extra={"key_name": api_key.name}
            )
            return False, None

        logger.debug(
            f"Valid API key authenticated: {api_key.name}",
            extra={"key_name": api_key.name}
        )

        return True, api_key

    def get_key_info(self, token: str) -> ApiKey | None:
        """Get information about an API key without validation.

        Args:
            token: API key string

        Returns:
            ApiKey object if found, None otherwise
        """
        return self._keys.get(token)

    def list_keys(self) -> list[ApiKey]:
        """List all registered API keys.

        Returns:
            List of all ApiKey objects
        """
        return list(self._keys.values())

    def count_keys(self) -> int:
        """Get count of registered API keys.

        Returns:
            Number of keys registered
        """
        return len(self._keys)

    @property
    def keys(self) -> dict[str, ApiKey]:
        """Get all registered API keys.

        Returns:
            Dictionary mapping key strings to ApiKey objects
        """
        return self._keys


class BearerTokenValidator:
    """Validates Bearer tokens from Authorization headers.

    This class extracts and validates Bearer tokens from HTTP
    Authorization headers following RFC 6750.

    Example:
        >>> validator = BearerTokenValidator()
        >>>
        >>> # Extract token from header
        >>> token = validator.extract_token("Bearer sk_test_12345")
        >>> print(token)  # "sk_test_12345"
        >>>
        >>> # Invalid format returns None
        >>> token = validator.extract_token("Invalid format")
        >>> print(token)  # None
    """

    @staticmethod
    def extract_token(authorization_header: str | None) -> str | None:
        """Extract Bearer token from Authorization header.

        Expected format: "Bearer <token>"

        Args:
            authorization_header: Value of Authorization header

        Returns:
            Extracted token if valid format, None otherwise
        """
        if not authorization_header:
            logger.debug("No Authorization header provided")
            return None

        # Split header into parts
        parts = authorization_header.split()

        # Check format: should be "Bearer <token>"
        if len(parts) != 2:
            logger.warning(
                "Malformed Authorization header - wrong number of parts",
                extra={"parts_count": len(parts)}
            )
            return None

        scheme, token = parts

        # Verify scheme is "Bearer" (case-insensitive)
        if scheme.lower() != "bearer":
            logger.warning(
                f"Invalid authorization scheme: {scheme}",
                extra={"scheme": scheme}
            )
            return None

        if not token:
            logger.warning("Empty token in Authorization header")
            return None

        logger.debug("Bearer token extracted successfully")
        return token

    @staticmethod
    def validate_format(authorization_header: str | None) -> tuple[bool, str]:
        """Validate Authorization header format without extracting token.

        Args:
            authorization_header: Value of Authorization header

        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if format is valid
            - error_message: Description of error if invalid, empty string if valid
        """
        if not authorization_header:
            return False, "Authorization header missing"

        parts = authorization_header.split()

        if len(parts) != 2:
            return False, "Invalid Authorization header format - expected 'Bearer <token>'"

        scheme, token = parts

        if scheme.lower() != "bearer":
            return False, f"Invalid authorization scheme '{scheme}' - expected 'Bearer'"

        if not token:
            return False, "Authorization token is empty"

        return True, ""


__all__ = [
    "ApiKey",
    "ApiKeyManager",
    "BearerTokenValidator",
]
