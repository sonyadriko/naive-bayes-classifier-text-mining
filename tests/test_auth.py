"""Tests for authentication endpoints."""

from fastapi.testclient import TestClient


def test_register_success(client: TestClient) -> None:
    """Test successful user registration."""
    response = client.post(
        "/api/v1/auth/register",
        data={
            "name": "John Doe",
            "email": "john@example.com",
            "password": "password123",
            "role": "user",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "access_token" in data["data"]
    assert data["data"]["user"]["email"] == "john@example.com"


def test_register_duplicate_email(client: TestClient) -> None:
    """Test registration with duplicate email."""
    # Register first user
    client.post(
        "/api/v1/auth/register",
        data={
            "name": "John Doe",
            "email": "john@example.com",
            "password": "password123",
            "role": "user",
        },
    )

    # Try to register with same email
    response = client.post(
        "/api/v1/auth/register",
        data={
            "name": "Jane Doe",
            "email": "john@example.com",
            "password": "password456",
            "role": "user",
        },
    )

    assert response.status_code == 400
    data = response.json()
    assert data["meta"]["status"] == "error"


def test_register_missing_fields(client: TestClient) -> None:
    """Test registration with missing fields."""
    response = client.post(
        "/api/v1/auth/register",
        data={
            "name": "John Doe",
            "email": "john@example.com",
            # Missing password
        },
    )

    assert response.status_code == 422  # Validation error


def test_login_success(client: TestClient) -> None:
    """Test successful login."""
    # Register user first
    client.post(
        "/api/v1/auth/register",
        data={
            "name": "John Doe",
            "email": "john@example.com",
            "password": "password123",
            "role": "user",
        },
    )

    # Login
    response = client.post(
        "/api/v1/auth/login",
        data={
            "email": "john@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "access_token" in data["data"]


def test_login_invalid_credentials(client: TestClient) -> None:
    """Test login with invalid credentials."""
    response = client.post(
        "/api/v1/auth/login",
        data={
            "email": "nonexistent@example.com",
            "password": "wrongpassword",
        },
    )

    assert response.status_code == 401
    data = response.json()
    assert data["meta"]["status"] == "error"


def test_login_wrong_password(client: TestClient) -> None:
    """Test login with wrong password."""
    # Register user first
    client.post(
        "/api/v1/auth/register",
        data={
            "name": "John Doe",
            "email": "john@example.com",
            "password": "password123",
            "role": "user",
        },
    )

    # Login with wrong password
    response = client.post(
        "/api/v1/auth/login",
        data={
            "email": "john@example.com",
            "password": "wrongpassword",
        },
    )

    assert response.status_code == 401
