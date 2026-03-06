"""Authentication Pydantic schemas."""

from typing import Optional

from pydantic import BaseModel, Field


class TokenPayload(BaseModel):
    """JWT token payload schema."""

    sub: str = Field(..., description="User ID (subject)")
    email: str = Field(..., description="User's email")
    role: str = Field(..., description="User's role")
    exp: Optional[int] = Field(None, description="Token expiration timestamp")


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    user: "UserInToken" = Field(..., description="User information")


class UserInToken(BaseModel):
    """User info embedded in token response."""

    id: int = Field(..., description="User's ID")
    email: str = Field(..., description="User's email")
    role: str = Field(..., description="User's role")


class RegisterRequest(BaseModel):
    """Schema for registration requests."""

    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., description="User's email")
    password: str = Field(..., min_length=8, max_length=100)
    role: str = Field(default="user", description="User's role")


class LoginResponse(BaseModel):
    """Schema for login response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer")
    user: "UserInfoInLogin"


class UserInfoInLogin(BaseModel):
    """User info in login response."""

    id: int
    email: str
    role: str
    name: str


class RefreshTokenRequest(BaseModel):
    """Schema for token refresh requests."""

    refresh_token: str = Field(..., description="Refresh token")
