"""Dependency injection providers for FastAPI routes."""

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import TokenManager
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import TokenPayload
from app.middleware.error_handler import AuthenticationError, NotFoundError

# Database dependency
DBSession = Annotated[Session, Depends(get_db)]


async def get_current_user(
    authorization: Annotated[str, Header(...)] = "",
    db: DBSession = None,
) -> User:
    """Get current authenticated user from JWT token.

    Args:
        authorization: Authorization header containing Bearer token.
        db: Database session.

    Returns:
        Authenticated user.

    Raises:
        AuthenticationError: If token is invalid or user not found.
    """
    if not authorization.startswith("Bearer "):
        raise AuthenticationError("Invalid authentication credentials")

    token = authorization.split(" ")[1]
    payload = TokenManager.decode_access_token(token)

    if payload is None:
        raise AuthenticationError("Could not validate credentials")

    token_data = TokenPayload(**payload)
    user_repo = UserRepository(db)

    result = user_repo.get_by_id(int(token_data.sub))
    if result.is_err() or result.value is None:
        raise NotFoundError("User not found")

    return result.value


# Type aliases for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentActiveUser = Annotated[User, Depends(get_current_user)]
