"""Security utilities for authentication and password hashing."""

import bcrypt
from datetime import datetime, timedelta
from typing import Any

from jose import JWTError, jwt

from app.core.config import get_settings

settings = get_settings()


class PasswordManager:
    """Manager for password hashing and verification."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt.

        Args:
            password: Plain text password.

        Returns:
            Hashed password.
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.

        Args:
            plain_password: Plain text password to verify.
            hashed_password: Hashed password to compare against.

        Returns:
            True if password matches, False otherwise.
        """
        return bcrypt.checkpw(
            plain_password.encode('utf-8'),
            hashed_password.encode('utf-8')
        )


class TokenManager:
    """Manager for JWT token creation and verification."""

    @staticmethod
    def create_access_token(
        data: dict[str, Any],
        expires_delta: timedelta | None = None,
    ) -> str:
        """Create a JWT access token.

        Args:
            data: Payload data to encode in the token.
            expires_delta: Optional expiration time delta.

        Returns:
            Encoded JWT token string.
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.access_token_expire_minutes
            )

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, settings.secret_key, algorithm=settings.algorithm
        )
        return encoded_jwt

    @staticmethod
    def decode_access_token(token: str) -> dict[str, Any] | None:
        """Decode and verify a JWT access token.

        Args:
            token: JWT token string to decode.

        Returns:
            Decoded token payload if valid, None otherwise.
        """
        try:
            payload = jwt.decode(
                token, settings.secret_key, algorithms=[settings.algorithm]
            )
            return payload
        except JWTError:
            return None


def create_token_response(user_id: int, email: str, role: str) -> dict[str, Any]:
    """Create a token response for authenticated user.

    Args:
        user_id: User's ID.
        email: User's email.
        role: User's role.

    Returns:
        Dictionary containing access token and user info.
    """
    access_token = TokenManager.create_access_token(
        data={"sub": str(user_id), "email": email, "role": role}
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user_id, "email": email, "role": role},
    }
