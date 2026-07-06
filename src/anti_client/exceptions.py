class AgentAPIError(Exception):
    """Base exception for API errors."""

class AuthError(AgentAPIError):
    """Exception raised for authorization errors (e.g., invalid token)."""

class ModelNotFoundError(AgentAPIError):
    """Exception raised when a requested model is not found or is unavailable."""

class RateLimitError(AgentAPIError):
    """Exception raised when the rate limit is exceeded (HTTP 429)."""

class ToolExecutionError(AgentAPIError):
    """Exception raised when a tool (function) execution fails."""
