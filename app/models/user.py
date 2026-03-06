"""SQLAlchemy User model."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


class User(Base):
    """User model for authentication and authorization.

    Attributes:
        id: Primary key.
        name: User's full name.
        email: User's email address (unique).
        password: Hashed password.
        role: User's role (e.g., admin, user).
        is_active: Whether the user account is active.
        created_at: Account creation timestamp.
        updated_at: Last update timestamp.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), nullable=False, default="user")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        """Return string representation of User."""
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

    def to_dict(self) -> dict:
        """Convert User model to dictionary (excludes password).

        Returns:
            Dictionary representation of user.
        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
