"""Application-level errors shared by the evaluation service."""


class ApplicationError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ApplicationError):
    """Raised when configuration is missing or invalid."""


class ValidationError(ApplicationError):
    """Raised when imported or submitted data fails validation."""


class NotFoundError(ApplicationError):
    """Raised when a requested record or asset does not exist."""

