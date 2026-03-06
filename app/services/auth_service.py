"""Authentication service with business logic.

Handles user authentication, registration, and token management.
"""

from sqlalchemy.orm import Session

from app.core.security import PasswordManager, TokenManager, create_token_response
from app.middleware.error_handler import AuthenticationError, ConflictError
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.auth import LoginResponse, RegisterRequest
from app.utils.response import Err, Ok, Result


class AuthService:
    """Service for authentication operations.

    Handles login, registration, and token validation.
    """

    def __init__(self, db: Session) -> None:
        """Initialize auth service.

        Args:
            db: Database session.
        """
        self.user_repo = UserRepository(db)

    def register(self, request: RegisterRequest) -> Result[LoginResponse, str]:
        """Register a new user.

        Args:
            request: Registration request data.

        Returns:
            Ok(LoginResponse) if successful, Err with message if failed.
        """
        # Check if email already exists
        email_check = self.user_repo.email_exists(request.email)
        if email_check.is_err():
            return Err(f"Failed to check email: {email_check.error}")

        if email_check.value:
            return Err("Email already registered")

        # Hash password
        hashed_password = PasswordManager.hash_password(request.password)

        # Create user
        create_result = self.user_repo.create(
            name=request.name,
            email=request.email,
            password=hashed_password,
            role=request.role,
        )

        if create_result.is_err():
            return Err(f"Failed to create user: {create_result.error}")

        user = create_result.value

        # Generate token response
        token_response = create_token_response(user.id, user.email, user.role)

        return Ok(
            LoginResponse(
                access_token=token_response["access_token"],
                token_type=token_response["token_type"],
                user={
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "name": user.name,
                },
            )
        )

    def login(self, email: str, password: str) -> Result[LoginResponse, str]:
        """Authenticate user with email and password.

        Args:
            email: User's email address.
            password: User's plain text password.

        Returns:
            Ok(LoginResponse) if successful, Err with message if failed.
        """
        # Find user by email
        user_result = self.user_repo.find_by_email(email)

        if user_result.is_err():
            return Err("Invalid email or password")

        user = user_result.value

        if user is None:
            return Err("Invalid email or password")

        # Verify password
        if not PasswordManager.verify_password(password, user.password):
            return Err("Invalid email or password")

        # Check if user is active
        if not user.is_active:
            return Err("User account is inactive")

        # Generate token response
        token_response = create_token_response(user.id, user.email, user.role)

        return Ok(
            LoginResponse(
                access_token=token_response["access_token"],
                token_type=token_response["token_type"],
                user={
                    "id": user.id,
                    "email": user.email,
                    "role": user.role,
                    "name": user.name,
                },
            )
        )

    def validate_token(self, token: str) -> Result[dict, str]:
        """Validate JWT token and extract payload.

        Args:
            token: JWT token string.

        Returns:
            Ok(payload) if valid, Err with message if invalid.
        """
        payload = TokenManager.decode_access_token(token)

        if payload is None:
            return Err("Invalid or expired token")

        # Verify user still exists
        try:
            user_id = int(payload.get("sub", 0))
            user_result = self.user_repo.get_by_id(user_id)

            if user_result.is_err() or user_result.value is None:
                return Err("User not found")

            return Ok(payload)

        except (ValueError, TypeError):
            return Err("Invalid token payload")

    def change_password(
        self,
        user_id: int,
        old_password: str,
        new_password: str,
    ) -> Result[bool, str]:
        """Change user password.

        Args:
            user_id: User's ID.
            old_password: Current password.
            new_password: New password.

        Returns:
            Ok(True) if successful, Err with message if failed.
        """
        # Get user
        user_result = self.user_repo.get_by_id(user_id)
        if user_result.is_err():
            return Err("User not found")

        user = user_result.value

        # Verify old password
        if not PasswordManager.verify_password(old_password, user.password):
            return Err("Current password is incorrect")

        # Hash new password
        hashed_password = PasswordManager.hash_password(new_password)

        # Update password
        update_result = self.user_repo.update(user_id, password=hashed_password)

        if update_result.is_err():
            return Err(f"Failed to update password: {update_result.error}")

        return Ok(True)
