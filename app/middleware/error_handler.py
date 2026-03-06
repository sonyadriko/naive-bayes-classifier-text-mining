"""Global error handling middleware and custom exceptions."""

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from app.utils.response import ApiResponse


class APIError(Exception):
    """Base API exception."""

    def __init__(
        self,
        message: str,
        code: str = "API_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error message.
            code: Error code identifier.
            status_code: HTTP status code.
            details: Additional error details.
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(APIError):
    """Validation error exception."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize validation error.

        Args:
            message: Validation error message.
            details: Additional validation details.
        """
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class NotFoundError(APIError):
    """Resource not found error exception."""

    def __init__(
        self,
        message: str = "Resource not found",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize not found error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(
            message=message,
            code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class AuthenticationError(APIError):
    """Authentication error exception."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authentication error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details=details,
        )


class AuthorizationError(APIError):
    """Authorization error exception."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize authorization error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(
            message=message,
            code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details=details,
        )


class ConflictError(APIError):
    """Conflict error exception (e.g., duplicate resource)."""

    def __init__(
        self,
        message: str = "Resource conflict",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize conflict error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=status.HTTP_409_CONFLICT,
            details=details,
        )


class BusinessLogicError(APIError):
    """Business logic error exception."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize business logic error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(
            message=message,
            code="BUSINESS_LOGIC_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions.

    Args:
        request: FastAPI request.
        exc: APIError exception.

    Returns:
        JSONResponse with error details.
    """
    return JSONResponse(
        content=ApiResponse.error(
            message=exc.message,
            code=exc.code,
            status_code=exc.status_code,
            details=exc.details,
        ),
        status_code=exc.status_code,
    )


async def validation_error_handler(
    request: Request,
    exc: PydanticValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: FastAPI request.
        exc: Pydantic ValidationError.

    Returns:
        JSONResponse with validation error details.
    """
    errors = {}
    for error in exc.errors():
        loc = ".".join(str(x) for x in error["loc"])
        errors[loc] = error["msg"]

    return JSONResponse(
        content=ApiResponse.error(
            message="Validation failed",
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"fields": errors},
        ),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic exceptions.

    Args:
        request: FastAPI request.
        exc: Exception.

    Returns:
        JSONResponse with error details.
    """
    return JSONResponse(
        content=ApiResponse.error(
            message="An unexpected error occurred",
            code="INTERNAL_SERVER_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"detail": str(exc)} if _is_debug() else None,
        ),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _is_debug() -> bool:
    """Check if application is in debug mode."""
    from app.core.config import get_settings

    return get_settings().debug


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers with the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(PydanticValidationError, validation_error_handler)
    app.add_exception_handler(Exception, generic_error_handler)
