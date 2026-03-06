"""Standardized API response builder.

All API responses follow this structure:
{
    "data": T | None,
    "meta": {
        "status": "success" | "error",
        "message": str,
        "timestamp": str,
        ...
    }
}
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

from fastapi import HTTPException
from fastapi.responses import JSONResponse

T = TypeVar("T")
E = TypeVar("E")


class ApiResponse(Generic[T]):
    """Standardized API response builder.

    Provides static methods for building consistent API responses
    across all endpoints following the DRY principle.
    """

    @staticmethod
    def success(
        data: T,
        message: str = "Success",
        status_code: int = 200,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a success response.

        Args:
            data: Response payload data.
            message: Success message.
            status_code: HTTP status code.
            meta: Optional additional metadata.

        Returns:
            Response dictionary with data and meta.
        """
        response_meta = {
            "status": "success",
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if meta:
            response_meta.update(meta)

        return {
            "data": data,
            "meta": response_meta,
        }

    @staticmethod
    def error(
        message: str,
        code: str = "ERROR",
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an error response.

        Args:
            message: Error description.
            code: Error code identifier.
            status_code: HTTP status code.
            details: Optional additional error details.

        Returns:
            Response dictionary with null data and error meta.
        """
        response_meta = {
            "status": "error",
            "message": message,
            "code": code,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        if details:
            response_meta["details"] = details

        return {
            "data": None,
            "meta": response_meta,
        }

    @staticmethod
    def paginated(
        data: list[T],
        total: int,
        page: int,
        page_size: int,
        message: str = "Success",
    ) -> dict[str, Any]:
        """Build a paginated response.

        Args:
            data: List of items.
            total: Total number of items.
            page: Current page number (1-indexed).
            page_size: Number of items per page.
            message: Success message.

        Returns:
            Response dictionary with pagination metadata.
        """
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 1

        return {
            "data": data,
            "meta": {
                "status": "success",
                "message": message,
                "pagination": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        }


class APIResponse(JSONResponse):
    """JSONResponse subclass for API responses.

    Usage:
        ```python
        @router.get("/users")
        async def get_users() -> APIResponse:
            return APIResponse(
                ApiResponse.success(data=users, message="Users retrieved")
            )
        ```
    """

    def __init__(
        self,
        content: dict[str, Any],
        status_code: int = 200,
    ) -> None:
        """Initialize API response.

        Args:
            content: Response content from ApiResponse builder.
            status_code: HTTP status code.
        """
        super().__init__(content=content, status_code=status_code)


# Convenience functions for raising HTTP exceptions with standardized responses
def raise_http_exception(
    message: str,
    code: str = "ERROR",
    status_code: int = 400,
    details: dict[str, Any] | None = None,
) -> None:
    """Raise HTTPException with standardized error response.

    Args:
        message: Error message.
        code: Error code.
        status_code: HTTP status code.
        details: Additional error details.

    Raises:
        HTTPException: With formatted error response.
    """
    raise HTTPException(
        status_code=status_code,
        detail=ApiResponse.error(message=message, code=code, details=details),
    )


class Result(Generic[T, E]):
    """Result type for error handling without exceptions.

    Represents either a success (Ok) or an error (Err).

    Usage:
        ```python
        def get_user(id: int) -> Result[User, str]:
            user = db.query(User).filter_by(id=id).first()
            if not user:
                return Err("User not found")
            return Ok(user)

        # Usage
        result = get_user(1)
        if isinstance(result, Err):
            return APIResponse(ApiResponse.error(result.error))
        user = result.value
        ```
    """

    def __init__(self, value: T | None, error: E | None) -> None:
        """Initialize Result.

        Args:
            value: Success value (None if error).
            error: Error value (None if success).
        """
        self._value = value
        self._error = error

    @property
    def value(self) -> T:
        """Get success value.

        Returns:
            Success value.

        Raises:
            ValueError: If result is an error.
        """
        if self.is_err():
            raise ValueError("Cannot get value from error result")
        return self._value  # type: ignore

    @property
    def error(self) -> E:
        """Get error value.

        Returns:
            Error value.

        Raises:
            ValueError: If result is a success.
        """
        if self.is_ok():
            raise ValueError("Cannot get error from success result")
        return self._error  # type: ignore

    def is_ok(self) -> bool:
        """Check if result is success."""
        return self._error is None

    def is_err(self) -> bool:
        """Check if result is error."""
        return self._error is not None

    def map(self, fn) -> "Result":  # type: ignore
        """Map success value through function.

        Args:
            fn: Function to apply to success value.

        Returns:
            New Result with mapped value or error.
        """
        if self.is_err():
            return self
        return Ok(fn(self._value))

    def and_then(self, fn) -> "Result":  # type: ignore
        """Chain Result-returning functions.

        Args:
            fn: Function that returns a Result.

        Returns:
            Result from fn or error.
        """
        if self.is_err():
            return self
        return fn(self._value)


class Ok(Result[T, E]):
    """Success variant of Result."""

    def __init__(self, value: T) -> None:
        """Initialize Ok result.

        Args:
            value: Success value.
        """
        super().__init__(value, None)


class Err(Result[T, E]):
    """Error variant of Result."""

    def __init__(self, error: E) -> None:
        """Initialize Err result.

        Args:
            error: Error value.
        """
        super().__init__(None, error)
