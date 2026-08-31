"""Custom exception classes for anti-client."""

from __future__ import annotations

from typing import Any


class AgentAPIError(Exception):
    """Base exception for all anti-client API and Agent errors."""


class AuthError(AgentAPIError):
    """Exception raised for authorization errors (e.g., missing, invalid, or expired tokens)."""


class ModelNotFoundError(AgentAPIError):
    """Exception raised when a requested model is not found or is unavailable."""


class RateLimitError(AgentAPIError):
    """Exception raised when an API rate limit is exceeded (HTTP 429).

    Attributes:
        quota_reset_delay (Optional[str]): Human-readable delay until quota resets (e.g., '120s').
        quota_reset_timestamp (Optional[str]): ISO 8601 timestamp when quota resets.
    """

    def __init__(
        self,
        message: str,
        quota_reset_delay: str | None = None,
        quota_reset_timestamp: str | None = None,
    ):
        """Initializes the RateLimitError.

        Args:
            message (str): Explanatory error message.
            quota_reset_delay (Optional[str], optional): Delay until quota refresh. Defaults to None.
            quota_reset_timestamp (Optional[str], optional): Timestamp of quota refresh. Defaults to None.
        """
        super().__init__(message)
        self.quota_reset_delay = quota_reset_delay
        self.quota_reset_timestamp = quota_reset_timestamp


class ToolExecutionError(AgentAPIError):
    """Exception raised when a tool (function) execution fails."""


class AgentMaxStepsError(AgentAPIError):
    """Exception raised when the agent exceeds the maximum allowed steps while executing tools.

    Attributes:
        last_response (Optional[Any]): The last ChatResponse generated before exceeding steps.
        steps_taken (int): Number of execution steps taken before failure.
    """

    def __init__(
        self,
        message: str,
        last_response: Any | None = None,
        steps_taken: int = 0,
    ):
        """Initializes the AgentMaxStepsError.

        Args:
            message (str): Explanatory error message.
            last_response (Optional[Any], optional): The last generated response. Defaults to None.
            steps_taken (int, optional): Number of steps executed. Defaults to 0.
        """
        super().__init__(message)
        self.last_response = last_response
        self.steps_taken = steps_taken
