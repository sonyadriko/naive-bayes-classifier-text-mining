"""User management API endpoints.

Handles user CRUD operations.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.user import UserUpdate
from app.services.user_service import UserService
from app.utils.response import ApiResponse

router = APIRouter()

# Type aliases
DBSession = Annotated[Session, Depends(get_db)]


def get_user_service(db: DBSession) -> UserService:
    """Get user service instance."""
    return UserService(db)


UserService = Annotated[UserService, Depends(get_user_service)]


def get_current_user_dependency(
    authorization: Annotated[str, Depends(lambda: None)] = "",
    db: DBSession = None,
):
    """Get current authenticated user from JWT token."""
    from dependencies import get_current_user
    return get_current_user(authorization, db)


CurrentUser = Annotated["User", Depends(get_current_user_dependency)]


@router.get(
    "",
    summary="List all users",
    description="Retrieve a paginated list of all users",
)
async def list_users(
    current_user: CurrentUser,
    user_service: UserService,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 10,
) -> JSONResponse:
    """List all users with pagination.

    Args:
        page: Page number (1-indexed).
        page_size: Number of items per page.
        current_user: Current authenticated user.
        user_service: User service instance.

    Returns:
        JSONResponse with paginated user list.
    """
    skip = (page - 1) * page_size

    result = user_service.list_users(skip=skip, limit=page_size)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    users, total = result.value
    user_responses = [u.to_dict() for u in users]

    return JSONResponse(
        content=ApiResponse.paginated(
            data=user_responses,
            total=total,
            page=page,
            page_size=page_size,
            message="Users retrieved successfully",
        ),
    )


@router.get(
    "/me",
    summary="Get current user",
    description="Get information about the currently authenticated user",
)
async def get_current_user_info(
    current_user: CurrentUser,
) -> JSONResponse:
    """Get current user info.

    Args:
        current_user: Current authenticated user.

    Returns:
        JSONResponse with user data.
    """
    return JSONResponse(
        content=ApiResponse.success(
            data=current_user.to_dict(),
            message="User retrieved successfully",
        ),
    )


@router.get(
    "/{user_id}",
    summary="Get user by ID",
    description="Retrieve a specific user by their ID",
)
async def get_user(
    user_id: int,
    current_user: CurrentUser,
    user_service: UserService,
) -> JSONResponse:
    """Get user by ID.

    Args:
        user_id: User's ID.
        current_user: Current authenticated user.
        user_service: User service instance.

    Returns:
        JSONResponse with user data.
    """
    result = user_service.get_user(user_id)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, status_code=404),
            status_code=404,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value.to_dict(),
            message="User retrieved successfully",
        ),
    )


@router.put(
    "/{user_id}",
    summary="Update user",
    description="Update user information",
)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: CurrentUser,
    user_service: UserService,
) -> JSONResponse:
    """Update user.

    Args:
        user_id: User's ID.
        user_data: Update data.
        current_user: Current authenticated user.
        user_service: User service instance.

    Returns:
        JSONResponse with updated user data.
    """
    result = user_service.update_user(user_id, user_data)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value.to_dict(),
            message="User updated successfully",
        ),
    )


@router.delete(
    "/{user_id}",
    summary="Delete user",
    description="Delete a user account",
)
async def delete_user(
    user_id: int,
    current_user: CurrentUser,
    user_service: UserService,
) -> JSONResponse:
    """Delete user.

    Args:
        user_id: User's ID.
        current_user: Current authenticated user.
        user_service: User service instance.

    Returns:
        JSONResponse confirming deletion.
    """
    result = user_service.delete_user(user_id)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"deleted": True},
            message="User deleted successfully",
        ),
    )
