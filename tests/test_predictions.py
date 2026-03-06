"""Tests for prediction endpoints."""

from fastapi.testclient import TestClient


def test_predict_success(client: TestClient, sample_excel_file, sample_prediction_data) -> None:
    """Test successful prediction."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Make prediction
    response = client.post(
        "/api/v1/predictions/predict",
        json=sample_prediction_data,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "predicted_class" in data["data"]
    assert "posteriors" in data["data"]
    assert "priors" in data["data"]


def test_predict_without_data(client: TestClient, sample_prediction_data) -> None:
    """Test prediction without training data."""
    response = client.post(
        "/api/v1/predictions/predict",
        json=sample_prediction_data,
    )

    assert response.status_code == 400
    data = response.json()
    assert data["meta"]["status"] == "error"


def test_get_model_info(client: TestClient, sample_excel_file) -> None:
    """Test getting model information."""
    # Upload file first to train model
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Get model info
    response = client.get("/api/v1/predictions/model/info")

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "is_trained" in data["data"]
    assert data["data"]["is_trained"] is True


def test_clear_model_cache(client: TestClient, sample_excel_file) -> None:
    """Test clearing model cache."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Clear cache
    response = client.post("/api/v1/predictions/model/cache/clear")

    assert response.status_code == 200
    data = response.json()
    assert data["data"]["cached"] is False


def test_evaluate_model(client: TestClient, sample_excel_file) -> None:
    """Test model evaluation."""
    # Upload file first
    with open(sample_excel_file, "rb") as f:
        client.post(
            "/api/v1/data/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Evaluate model
    response = client.post(
        "/api/v1/evaluation/confusion-matrix",
        json={"test_size": 0.2},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["meta"]["status"] == "success"
    assert "confusion_matrix" in data["data"]
    assert "accuracy" in data["data"]
    assert "precision" in data["data"]
    assert "recall" in data["data"]
    assert "f1_score" in data["data"]
