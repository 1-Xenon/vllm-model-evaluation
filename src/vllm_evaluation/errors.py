"""Application-level errors shared by the evaluation service."""

from dataclasses import dataclass


class ApplicationError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(ApplicationError):
    """Raised when configuration is missing or invalid."""


class ValidationError(ApplicationError):
    """Raised when imported or submitted data fails validation."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One structured validation problem with a stable location and code."""

    location: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "location": self.location,
            "code": self.code,
            "message": self.message,
        }


class ManifestValidationError(ValidationError):
    """Raised when a task manifest or referenced bundle asset is invalid."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = tuple(issues)
        summary = "; ".join(f"{issue.location}: {issue.message}" for issue in self.issues)
        super().__init__(summary or "manifest validation failed")


class NotFoundError(ApplicationError):
    """Raised when a requested record or asset does not exist."""
