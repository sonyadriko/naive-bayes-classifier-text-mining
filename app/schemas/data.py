"""Data processing Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field


class LabelInfo(BaseModel):
    """Schema for label encoding information."""

    encoded_values: dict[int, str] = Field(
        ...,
        description="Mapping of encoded values to original labels",
    )
    classes: list[str] = Field(..., description="List of unique class labels")


class LabelsResponse(BaseModel):
    """Schema for labels response."""

    labels: dict[str, LabelInfo] = Field(
        ...,
        description="Label encoding info for each feature column",
    )


class DataUploadResponse(BaseModel):
    """Schema for data upload response."""

    message: str = Field(..., description="Upload result message")
    file_path: str = Field(..., description="Path to saved file")
    rows: int = Field(..., description="Number of rows processed")
    columns: int = Field(..., description="Number of columns processed")


class DataConvertRequest(BaseModel):
    """Schema for data conversion request."""

    input_data: dict[str, Any] = Field(
        ...,
        description="Input data to convert using label encoders",
    )


class DataConvertResponse(BaseModel):
    """Schema for data conversion response."""

    converted_data: dict[str, Any] = Field(
        ...,
        description="Converted data with encoded values",
    )


class DataRow(BaseModel):
    """Schema for a single data row."""

    jenisKelamin: Any
    organisasi: Any
    ekstrakurikuler: Any
    sertifikasiProfesi: Any
    nilaiAkhir: Any
    tempatMagang: Any
    tempatKerja: Any
    Durasi_Mendapat_Kerja: Any = Field(..., alias="Durasi Mendapat Kerja")

    model_config = {"populate_by_name": True}


class DataReadResponse(BaseModel):
    """Schema for data read response."""

    data: list[DataRow] = Field(..., description="Data rows from Excel file")
    total: int = Field(..., description="Total number of rows")


class FeatureInfo(BaseModel):
    """Schema for feature information."""

    name: str = Field(..., description="Feature name")
    type: str = Field(..., description="Feature data type")
    unique_values: int = Field(..., description="Number of unique values")
    sample_values: list[Any] = Field(..., description="Sample values")


class DataInfoResponse(BaseModel):
    """Schema for dataset information response."""

    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    target_column: str = Field(..., description="Target column name")
    features: list[FeatureInfo] = Field(..., description="Feature information")
