"""Pytest configuration and fixtures."""

import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.core.database import get_db
from main import app

# Test settings
os.environ["ENVIRONMENT"] = "testing"
os.environ["DB_NAME"] = "test_naive_bayes"
os.environ["SECRET_KEY"] = "test-secret-key-for-testing-only"

settings = get_settings()

# Test database engine
TEST_DATABASE_URL = settings.database_url
test_engine = create_engine(TEST_DATABASE_URL, pool_pre_ping=True)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db() -> Generator[Session, None, None]:
    """Create a test database session.

    Yields:
        Test database session.
    """
    # Create all tables
    from app.models.user import Base, User

    Base.metadata.create_all(bind=test_engine)

    session = TestSessionLocal()
    try:
        yield session
    finally:
        session.close()
        # Drop all tables after test
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db: Session) -> Generator[TestClient, None, None]:
    """Create a test client.

    Args:
        db: Test database session.

    Yields:
        Test client with database override.
    """

    def override_get_db():
        try:
            yield db
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_excel_file() -> Generator[Path, None, None]:
    """Create a sample Excel file for testing.

    Yields:
        Path to temporary Excel file.
    """
    data = pd.DataFrame(
        {
            "jenisKelamin": ["L", "P", "L", "P", "L"],
            "organisasi": ["Ya", "Tidak", "Ya", "Ya", "Tidak"],
            "ekstrakurikuler": ["Ya", "Ya", "Tidak", "Ya", "Tidak"],
            "sertifikasiProfesi": ["Ya", "Tidak", "Ya", "Tidak", "Ya"],
            "nilaiAkhir": [85, 78, 90, 82, 75],
            "tempatMagang": ["Perusahaan A", "Perusahaan B", "Perusahaan A", "Perusahaan C", "Perusahaan B"],
            "tempatKerja": ["Perusahaan A", "Perusahaan B", "Perusahaan A", "Perusahaan C", "Perusahaan B"],
            "Durasi Mendapat Kerja": ["< 3 bulan", "3-6 bulan", "< 3 bulan", "> 6 bulan", "3-6 bulan"],
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        data.to_excel(f.name, index=False)
        yield Path(f.name)

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def auth_headers(client: TestClient) -> dict[str, str]:
    """Get authentication headers for a test user.

    Args:
        client: Test client.

    Returns:
        Dictionary with Authorization header.
    """
    # Register a test user
    client.post(
        "/api/v1/auth/register",
        data={
            "name": "Test User",
            "email": "test@example.com",
            "password": "testpassword123",
            "role": "user",
        },
    )

    # Login and get token
    response = client.post(
        "/api/v1/auth/login",
        data={
            "email": "test@example.com",
            "password": "testpassword123",
        },
    )

    token = response.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_prediction_data() -> dict:
    """Get sample prediction data.

    Returns:
        Dictionary with feature values.
    """
    return {
        "jenisKelamin": "L",
        "organisasi": "Ya",
        "ekstrakurikuler": "Ya",
        "sertifikasiProfesi": "Ya",
        "nilaiAkhir": 85,
        "tempatMagang": "Perusahaan A",
        "tempatKerja": "Perusahaan A",
    }
