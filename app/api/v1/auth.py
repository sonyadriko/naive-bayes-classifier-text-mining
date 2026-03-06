"""Authentication API endpoints.

Handles user login and registration.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Form, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.auth import LoginResponse, RegisterRequest
from app.services.auth_service import AuthService
from app.utils.response import ApiResponse

router = APIRouter()

# Type aliases
DBSession = Annotated[Session, Depends(get_db)]


def get_auth_service(db: DBSession) -> AuthService:
    """Get auth service instance."""
    return AuthService(db)


AuthService = Annotated[AuthService, Depends(get_auth_service)]


@router.post(
    "/register",
    summary="Register a new user",
    description="Create a new user account with email and password",
    status_code=status.HTTP_201_CREATED,
)
async def register(
    auth_service: AuthService,
    name: str = Form(..., description="User's full name"),
    email: str = Form(..., description="User's email address"),
    password: str = Form(..., min_length=8, description="User's password (min 8 characters)"),
    role: str = Form(default="user", description="User's role"),
) -> JSONResponse:
    """Register a new user.

    Args:
        name: User's full name.
        email: User's email address.
        password: User's password.
        role: User's role (default: user).
        auth_service: Auth service instance.

    Returns:
        JSONResponse with user data and access token.
    """
    request = RegisterRequest(
        name=name,
        email=email,
        password=password,
        role=role,
    )

    result = auth_service.register(request)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value.model_dump(),
            message="User registered successfully",
        ),
        status_code=status.HTTP_201_CREATED,
    )


@router.post(
    "/login",
    summary="Login user",
    description="Authenticate user with email and password",
)
async def login(
    auth_service: AuthService,
    email: str = Form(..., description="User's email address"),
    password: str = Form(..., description="User's password"),
) -> JSONResponse:
    """Login user.

    Args:
        email: User's email address.
        password: User's password.
        auth_service: Auth service instance.

    Returns:
        JSONResponse with access token and user info.
    """
    result = auth_service.login(email, password)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value.model_dump(),
            message="Login successful",
        ),
        status_code=status.HTTP_200_OK,
    )
