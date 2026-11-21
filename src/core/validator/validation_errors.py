from __future__ import annotations


class ValidationError(Exception):
    """Raised when a job shop schedule violates one or more constraints.

    Attributes:
        message: High-level summary of the validation failure.
        errors: Detailed list of constraint violation messages.
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors: list[str] = list(errors) if errors is not None else [message]
