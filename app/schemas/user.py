"""User Pydantic schemas for request/response validation."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


class BaseUserSchema(BaseModel):
    """Base user schema with common fields."""

    name: str = Field(..., min_length=1, max_length=255, description="User's full name")
    email: EmailStr = Field(..., description="User's email address")
    role: str = Field(..., min_length=1, max_length=50, description="User's role")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate role is lowercase."""
        return v.lower()


class UserCreate(BaseUserSchema):
    """Schema for user creation (register) requests."""

    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="User's password (min 8 characters)",
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets minimum requirements."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        return v


class UserUpdate(BaseModel):
    """Schema for user update requests."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    role: Optional[str] = Field(None, min_length=1, max_length=50)
    is_active: Optional[bool] = None


class UserResponse(BaseUserSchema):
    """Schema for user response."""

    id: int = Field(..., description="User's ID")
    is_active: bool = Field(default=True, description="Whether user account is active")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    """Schema for user list response."""

    users: list[UserResponse]
    total: int
    page: int
    page_size: int


class UserLogin(BaseModel):
    """Schema for user login requests."""

    email: EmailStr = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


class UserChangePassword(BaseModel):
    """Schema for password change requests."""

    old_password: str = Field(..., description="Current password")
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="New password (min 8 characters)",
    )
