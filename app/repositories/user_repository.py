"""User repository with user-specific queries.

Extends BaseRepository with user-specific operations.
"""

from sqlalchemy.orm import Session

from app.models.user import User
from app.repositories.base import BaseRepository
from app.utils.response import Err, Ok, Result


class UserRepository(BaseRepository[User]):
    """Repository for User model with user-specific queries."""

    def __init__(self, session: Session) -> None:
        """Initialize user repository.

        Args:
            session: Database session.
        """
        super().__init__(User, session)

    def get_by_id(self, user_id: int) -> Result[User, str]:
        """Get user by ID.

        Args:
            user_id: User ID.

        Returns:
            Result containing user or error message.
        """
        return self.get(user_id)

    def find_by_email(self, email: str) -> Result[User | None, str]:
        """Find user by email address.

        Args:
            email: User email address.

        Returns:
            Result containing user or None if not found, or error message.
        """
        return self.find_one_by(email=email)

    def email_exists(self, email: str) -> Result[bool, str]:
        """Check if user with given email exists.

        Args:
            email: Email address to check.

        Returns:
            Result containing True if email exists, False otherwise, or error.
        """
        result = self.find_by_email(email)
        if result.is_err():
            return result
        return Ok(result.value is not None)

    def get_by_role(self, role: str, skip: int = 0, limit: int = 100) -> Result[list[User], str]:
        """Get users by role.

        Args:
            role: User role to filter by.
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            Result containing list of users or error message.
        """
        try:
            users = (
                self.session.query(User)
                .filter(User.role == role)
                .offset(skip)
                .limit(limit)
                .all()
            )
            return Ok(users)
        except Exception as e:
            return Err(f"Failed to get users by role: {str(e)}")
