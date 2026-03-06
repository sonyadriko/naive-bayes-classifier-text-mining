"""Tests for data processing endpoints."""

from fastapi.testclient import TestClient


def test_upload_file_success(client: TestClient, sample_excel_file) -> None:
    """Test successful file upload."""
    with open(sample_excel_file, "rb") as f:
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "rows" in data["data"]


def test_upload_invalid_format(client: TestClient) -> None:
    """Test upload with invalid file format."""
    response = client.post(
        "/api/v1/data/upload",
        files={"file": ("test.txt", b"some text content", "text/plain")},
    )

    assert response.status_code == 400


def test_get_labels_after_upload(client: TestClient, sample_excel_file) -> None:
    """Test getting labels after file upload."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Get labels
    response = client.get("/api/v1/data/labels")

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "labels" in data["data"]


def test_read_data_after_upload(client: TestClient, sample_excel_file) -> None:
    """Test reading data after file upload."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Read data
    response = client.get("/api/v1/data/read")

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "rows" in data["data"]
    assert len(data["data"]["rows"]) > 0


def test_get_data_info(client: TestClient, sample_excel_file) -> None:
    """Test getting dataset information."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Get data info
    response = client.get("/api/v1/data/info")

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "total_rows" in data["data"]
    assert "features" in data["data"]


def test_convert_data(client: TestClient, sample_excel_file) -> None:
    """Test converting categorical data."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Convert data
    response = client.post(
        "/api/v1/data/convert",
        json={
            "input_data": {
                "jenisKelamin": "L",
                "organisasi": "Ya",
                "ekstrakurikuler": "Ya",
                "sertifikasiProfesi": "Ya",
                "nilaiAkhir": 85,
                "tempatMagang": "Perusahaan A",
                "tempatKerja": "Perusahaan A",
            }
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
