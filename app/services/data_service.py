"""Data processing service.

Handles Excel file processing, data conversion, and reading.
"""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.middleware.error_handler import BusinessLogicError
from app.services.label_service import LabelService
from app.utils.response import Err, Ok, Result

settings = get_settings()

# Expected column headers
EXPECTED_COLUMNS = [
    "jenisKelamin",
    "organisasi",
    "ekstrakurikuler",
    "sertifikasiProfesi",
    "nilaiAkhir",
    "tempatMagang",
    "tempatKerja",
    "Durasi Mendapat Kerja",
]


class DataService:
    """Service for data processing operations.

    Handles file upload, data conversion, and reading operations.
    """

    def __init__(self) -> None:
        """Initialize data service."""
        self.label_service = LabelService()
        self.data_dir = Path(settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._current_file_hash: str | None = None

    def process_upload(self, file_content: bytes, filename: str) -> Result[dict[str, Any], str]:
        """Process uploaded Excel file.

        Args:
            file_content: File content as bytes.
            filename: Original filename.

        Returns:
            Ok(dict with file info) if successful, Err with message if failed.
        """
        try:
            # Validate file extension
            if not self._is_valid_file(filename):
                return Err("Invalid file format. Only .xls and .xlsx files are allowed")

            # Read Excel file
            data = pd.read_excel(file_content, header=0)

            # Validate columns
            if len(data.columns) != len(EXPECTED_COLUMNS):
                return Err(
                    f"Expected {len(EXPECTED_COLUMNS)} columns, got {len(data.columns)}"
                )

            # Rename columns to expected headers
            data.columns = EXPECTED_COLUMNS

            # Fit label encoders
            fit_result = self.label_service.fit_encoders(data)
            if fit_result.is_err():
                return Err(f"Failed to fit encoders: {fit_result.error}")

            # Save processed data
            file_hash = self._generate_file_hash(file_content)
            self._current_file_hash = file_hash

            processed_path = self.data_dir / "data.xlsx"
            data.to_excel(processed_path, index=False, sheet_name="Sheet1")

            # Save encoders
            self.label_service.save_encoders(file_hash)

            return Ok(
                {
                    "message": "File processed successfully",
                    "file_path": str(processed_path),
                    "rows": len(data),
                    "columns": len(data.columns),
                }
            )

        except Exception as e:
            return Err(f"Failed to process file: {str(e)}")

    def convert_data(self, input_data: dict[str, Any]) -> Result[dict[str, Any], str]:
        """Convert categorical data using label encoders.

        Args:
            input_data: Dictionary of feature names to values.

        Returns:
            Ok(converted_data) if successful, Err with message if failed.
        """
        if not self.label_service.has_encoders():
            # Try to load encoders from cache
            load_result = self.label_service.load_encoders(self._current_file_hash or "default")
            if load_result.is_err():
                return Err("No label encoders available. Please upload a file first.")

        return self.label_service.transform(input_data)

    def read_data(self) -> Result[list[dict[str, Any]], str]:
        """Read data from Excel file.

        Returns:
            Ok(list of data rows) if successful, Err with message if failed.
        """
        try:
            data_path = self.data_dir / "data.xlsx"

            if not data_path.exists():
                return Err("No data file found. Please upload a file first.")

            data = pd.read_excel(data_path)
            return Ok(data.to_dict(orient="records"))

        except Exception as e:
            return Err(f"Failed to read data: {str(e)}")

    def get_labels(self) -> Result[dict[str, dict[str, Any]], str]:
        """Get label encoder mappings.

        Returns:
            Ok(dict of label mappings) if successful, Err with message if failed.
        """
        if not self.label_service.has_encoders():
            # Try to load from cache
            load_result = self.label_service.load_encoders(self._current_file_hash or "default")
            if load_result.is_err():
                return Err("No label encoders available. Please upload a file first.")

        return self.label_service.get_mappings()

    def get_training_data(self) -> Result[tuple[pd.DataFrame, pd.Series], str]:
        """Get training data split into features and target.

        Returns:
            Ok((X, y)) if successful, Err with message if failed.
        """
        try:
            data_path = self.data_dir / "data.xlsx"

            if not data_path.exists():
                return Err("No training data found. Please upload a file first.")

            data = pd.read_excel(data_path)

            if settings.target_column not in data.columns:
                return Err(f"Target column '{settings.target_column}' not found in data")

            X = data.drop(columns=[settings.target_column])
            y = data[settings.target_column]

            return Ok((X, y))

        except Exception as e:
            return Err(f"Failed to load training data: {str(e)}")

    def get_data_info(self) -> Result[dict[str, Any], str]:
        """Get information about the dataset.

        Returns:
            Ok(dict with dataset info) if successful, Err with message if failed.
        """
        try:
            data_path = self.data_dir / "data.xlsx"

            if not data_path.exists():
                return Err("No data file found. Please upload a file first.")

            data = pd.read_excel(data_path)

            features = []
            for col in data.columns:
                if col != settings.target_column:
                    features.append(
                        {
                            "name": col,
                            "type": str(data[col].dtype),
                            "unique_values": int(data[col].nunique()),
                            "sample_values": data[col].head(3).tolist(),
                        }
                    )

            return Ok(
                {
                    "total_rows": len(data),
                    "total_columns": len(data.columns),
                    "target_column": settings.target_column,
                    "features": features,
                }
            )

        except Exception as e:
            return Err(f"Failed to get data info: {str(e)}")

    def _is_valid_file(self, filename: str) -> bool:
        """Check if file has valid extension.

        Args:
            filename: Name of the file.

        Returns:
            True if valid, False otherwise.
        """
        return any(filename.endswith(ext) for ext in settings.allowed_extensions)

    @staticmethod
    def _generate_file_hash(content: bytes) -> str:
        """Generate hash from file content.

        Args:
            content: File content as bytes.

        Returns:
            Hash string.
        """
        return hashlib.sha256(content).hexdigest()[:16]
