"""Prediction Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for prediction requests.

    Features for predicting job placement duration.
    """

    jenisKelamin: str | int = Field(..., description="Gender (L/P or encoded)")
    organisasi: str | int = Field(..., description="Organizational experience")
    ekstrakurikuler: str | int = Field(..., description="Extracurricular activities")
    sertifikasiProfesi: str | int = Field(..., description="Professional certification")
    nilaiAkhir: int | float = Field(..., description="Final grade")
    tempatMagang: str | int = Field(..., description="Internship location")
    tempatKerja: str | int = Field(..., description="Workplace location")


class LikelihoodDetail(BaseModel):
    """Schema for likelihood details."""

    jenisKelamin: float = Field(..., description="Likelihood for gender")
    organisasi: float = Field(..., description="Likelihood for organization")
    ekstrakurikuler: float = Field(..., description="Likelihood for extracurricular")
    sertifikasiProfesi: float = Field(..., description="Likelihood for certification")
    nilaiAkhir: float = Field(..., description="Likelihood for final grade")
    tempatMagang: float = Field(..., description="Likelihood for internship")
    tempatKerja: float = Field(..., description="Likelihood for workplace")


class PredictionResponse(BaseModel):
    """Schema for prediction response."""

    predicted_class: str | int = Field(..., description="Predicted class")
    posteriors: dict[str, float] = Field(..., description="Posterior probabilities")
    likelihoods: dict[str, dict[str, float]] = Field(..., description="Likelihood details")
    priors: dict[str, float] = Field(..., description="Prior probabilities")
    evidence: float = Field(..., description="Evidence (normalization factor)")


class EvaluationRequest(BaseModel):
    """Schema for model evaluation requests."""

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.9,
        description="Proportion of data for testing (0.1-0.9)",
    )


class ConfusionMatrixResponse(BaseModel):
    """Schema for confusion matrix evaluation response."""

    test_size: float = Field(..., description="Test data proportion")
    confusion_matrix: list[list[int]] = Field(..., description="Confusion matrix")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
