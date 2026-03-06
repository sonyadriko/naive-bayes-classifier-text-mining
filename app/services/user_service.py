"""User service with business logic.

Handles user CRUD operations.
"""

from typing import Optional

from sqlalchemy.orm import Session

from app.middleware.error_handler import NotFoundError
from app.models.user import User
from app.repositories.user_repository import UserRepository
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.utils.response import Err, Ok, Result


class UserService:
    """Service for user CRUD operations.

    Handles user creation, retrieval, update, and deletion.
    """

    def __init__(self, db: Session) -> None:
        """Initialize user service.

        Args:
            db: Database session.
        """
        self.user_repo = UserRepository(db)

    def get_user(self, user_id: int) -> Result[User, str]:
        """Get user by ID.

        Args:
            user_id: User's ID.

        Returns:
            Ok(User) if found, Err with message if not found.
        """
        return self.user_repo.get_by_id(user_id)

    def get_user_by_email(self, email: str) -> Result[Optional[User], str]:
        """Get user by email.

        Args:
            email: User's email address.

        Returns:
            Ok(User) if found, Ok(None) if not found, Err on failure.
        """
        return self.user_repo.find_by_email(email)

    def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> Result[tuple[list[User], int], str]:
        """List all users with pagination.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            Ok((users, total)) if successful, Err with message if failed.
        """
        users_result = self.user_repo.get_all(skip=skip, limit=limit)
        count_result = self.user_repo.count()

        if users_result.is_err():
            return Err(users_result.error)

        if count_result.is_err():
            return Err(count_result.error)

        return Ok((users_result.value, count_result.value))

    def create_user(self, user_data: UserCreate) -> Result[User, str]:
        """Create a new user.

        Args:
            user_data: User creation data.

        Returns:
            Ok(User) if successful, Err with message if failed.
        """
        # Check if email already exists
        email_check = self.user_repo.email_exists(user_data.email)
        if email_check.is_err():
            return Err(f"Failed to check email: {email_check.error}")

        if email_check.value:
            return Err("Email already registered")

        # Hash password
        from app.core.security import PasswordManager

        hashed_password = PasswordManager.hash_password(user_data.password)

        # Create user
        create_result = self.user_repo.create(
            name=user_data.name,
            email=user_data.email,
            password=hashed_password,
            role=user_data.role,
        )

        if create_result.is_err():
            return Err(f"Failed to create user: {create_result.error}")

        return Ok(create_result.value)

    def update_user(
        self,
        user_id: int,
        user_data: UserUpdate,
    ) -> Result[User, str]:
        """Update user information.

        Args:
            user_id: User's ID.
            user_data: User update data.

        Returns:
            Ok(User) if successful, Err with message if failed.
        """
        # Check if user exists
        exists_result = self.user_repo.exists(user_id)
        if exists_result.is_err():
            return Err(f"Failed to check user: {exists_result.error}")

        if not exists_result.value:
            return Err("User not found")

        # Prepare update data
        update_data = user_data.model_dump(exclude_unset=True)

        # If email is being updated, check for duplicates
        if "email" in update_data:
            existing_result = self.user_repo.find_by_email(update_data["email"])
            if existing_result.is_ok() and existing_result.value is not None:
                if existing_result.value.id != user_id:
                    return Err("Email already in use")

        # Update user
        update_result = self.user_repo.update(user_id, **update_data)

        if update_result.is_err():
            return Err(f"Failed to update user: {update_result.error}")

        return Ok(update_result.value)

    def delete_user(self, user_id: int) -> Result[bool, str]:
        """Delete a user.

        Args:
            user_id: User's ID.

        Returns:
            Ok(True) if successful, Err with message if failed.
        """
        # Check if user exists
        exists_result = self.user_repo.exists(user_id)
        if exists_result.is_err():
            return Err(f"Failed to check user: {exists_result.error}")

        if not exists_result.value:
            return Err("User not found")

        # Delete user
        delete_result = self.user_repo.delete(user_id)

        if delete_result.is_err():
            return Err(f"Failed to delete user: {delete_result.error}")

        return Ok(True)

    def to_response(self, user: User) -> UserResponse:
        """Convert User model to UserResponse schema.

        Args:
            user: User model.

        Returns:
            UserResponse schema.
        """
        return UserResponse.model_validate(user)
