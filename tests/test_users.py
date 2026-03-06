"""Tests for user management endpoints."""

from fastapi.testclient import TestClient


def test_get_current_user(client: TestClient, auth_headers: dict) -> None:
    """Test getting current user info."""
    response = client.get(
        "/api/v1/users/me",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert data["data"]["email"] == "test@example.com"


def test_get_current_user_unauthorized(client: TestClient) -> None:
    """Test getting current user without authentication."""
    response = client.get("/api/v1/users/me")

    assert response.status_code == 401


def test_list_users(client: TestClient, auth_headers: dict) -> None:
    """Test listing users."""
    response = client.get(
        "/api/v1/users",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "data" in data
    assert isinstance(data["data"], list)


def test_list_users_pagination(client: TestClient, auth_headers: dict) -> None:
    """Test listing users with pagination."""
    response = client.get(
        "/api/v1/users?page=1&page_size=10",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["pagination"]["page"] == 1
    assert data["meta"]["pagination"]["page_size"] == 10


def test_get_user_by_id(client: TestClient, auth_headers: dict) -> None:
    """Test getting user by ID."""
    # First get current user to get ID
    me_response = client.get("/api/v1/users/me", headers=auth_headers)
    user_id = me_response.json()["data"]["id"]

    # Get user by ID
    response = client.get(
        f"/api/v1/users/{user_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["id"] == user_id


def test_update_user(client: TestClient, auth_headers: dict) -> None:
    """Test updating user."""
    # Get user ID
    me_response = client.get("/api/v1/users/me", headers=auth_headers)
    user_id = me_response.json()["data"]["id"]

    # Update user
    response = client.put(
        f"/api/v1/users/{user_id}",
        headers=auth_headers,
        json={"name": "Updated Name"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["name"] == "Updated Name"


def test_delete_user(client: TestClient, auth_headers: dict) -> None:
    """Test deleting user."""
    # Create a new user to delete
    client.post(
        "/api/v1/auth/register",
        data={
            "name": "Delete Me",
            "email": "delete@example.com",
            "password": "password123",
            "role": "user",
        },
    )

    # Login as delete user
    login_response = client.post(
        "/api/v1/auth/login",
        data={
            "email": "delete@example.com",
            "password": "password123",
        },
    )
    token = login_response.json()["data"]["access_token"]
    delete_headers = {"Authorization": f"Bearer {token}"}

    # Get user ID
    me_response = client.get("/api/v1/users/me", headers=delete_headers)
    user_id = me_response.json()["data"]["id"]

    # Delete user
    response = client.delete(
        f"/api/v1/users/{user_id}",
        headers=delete_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["deleted"] is True
