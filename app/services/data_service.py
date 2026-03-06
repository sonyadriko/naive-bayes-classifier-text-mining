"""Data processing service.

Handles Excel file processing, data conversion, and reading.
Supports both categorical features (job placement) and text data (sentiment analysis).
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

# Expected column headers for categorical data
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

# Expected columns for text/sentiment data
SENTIMENT_COLUMNS = ["content", "score"]  # score is optional for auto-labeling


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
        """Read data from Excel file or reviews CSV.

        Returns:
            Ok(list of data rows) if successful, Err with message if failed.
        """
        try:
            # First try to read categorical data (data.xlsx)
            data_path = self.data_dir / "data.xlsx"

            # Fall back to reviews.csv if data.xlsx doesn't exist
            if not data_path.exists():
                data_path = self.data_dir / "reviews.csv"

            if not data_path.exists():
                return Err("No data file found. Please upload a file first.")

            # Read based on file extension
            if data_path.suffix == ".csv":
                data = pd.read_csv(data_path)
            else:
                data = pd.read_excel(data_path)

            # Replace NaN values with None for JSON serialization
            return Ok(data.where(pd.notnull(data), None).to_dict(orient="records"))

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
            # First try to read categorical data (data.xlsx)
            data_path = self.data_dir / "data.xlsx"

            # Fall back to reviews.csv if data.xlsx doesn't exist
            if not data_path.exists():
                data_path = self.data_dir / "reviews.csv"

            if not data_path.exists():
                return Err("No data file found. Please upload a file first.")

            # Read based on file extension
            if data_path.suffix == ".csv":
                data = pd.read_csv(data_path)
            else:
                data = pd.read_excel(data_path)

            # Check if this is sentiment data (has content/sentiment columns)
            is_sentiment_data = "content" in data.columns or "sentiment" in data.columns

            if is_sentiment_data:
                # Sentiment analysis data info
                sentiment_dist = {}
                if "sentiment" in data.columns:
                    # Drop NaN values before counting
                    sentiment_dist = data["sentiment"].dropna().value_counts().to_dict()

                return Ok(
                    {
                        "total_rows": len(data),
                        "total_columns": len(data.columns),
                        "data_type": "sentiment",
                        "sentiment_distribution": sentiment_dist,
                    }
                )
            else:
                # Categorical data info
                features = []
                for col in data.columns:
                    if col != settings.target_column:
                        # Replace NaN with None in sample values for JSON serialization
                        sample_values = data[col].head(3).where(pd.notnull(data[col].head(3)), None).tolist()
                        features.append(
                            {
                                "name": col,
                                "type": str(data[col].dtype),
                                "unique_values": int(data[col].nunique()),
                                "sample_values": sample_values,
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

    # Text/Sentiment Analysis Methods

    def process_text_upload(self, file_content: bytes, filename: str) -> Result[dict[str, Any], str]:
        """Process uploaded text review file (CSV or Excel).

        Args:
            file_content: File content as bytes.
            filename: Original filename.

        Returns:
            Ok(dict with file info) if successful, Err with message if failed.
        """
        try:
            # Validate file extension
            if not self._is_valid_text_file(filename):
                return Err("Invalid file format. Only .csv, .xls, and .xlsx files are allowed")

            # Read file based on extension
            if filename.endswith(".csv"):
                data = pd.read_csv(file_content, encoding="utf-8", on_bad_lines="skip")
            else:
                data = pd.read_excel(file_content, header=0)

            # Validate columns
            if "content" not in data.columns:
                return Err("Missing 'content' column. File must have a 'content' column with text data.")

            # Ensure score column exists (add if missing)
            if "score" not in data.columns:
                data["score"] = None

            # Clean data
            data = data.dropna(subset=["content"])
            data["content"] = data["content"].astype(str)

            # Save processed data
            processed_path = self.data_dir / "reviews.csv"
            data.to_csv(processed_path, index=False, encoding="utf-8")

            return Ok(
                {
                    "message": "File processed successfully",
                    "file_path": str(processed_path),
                    "rows": len(data),
                    "has_scores": data["score"].notna().any(),
                }
            )

        except Exception as e:
            return Err(f"Failed to process text file: {str(e)}")

    def load_text_dataset(self) -> Result[tuple[pd.DataFrame, pd.Series], str]:
        """Load text dataset for sentiment analysis.

        Returns:
            Ok((X, y)) where X is text content and y is sentiment labels,
            Err with message if failed.
        """
        try:
            data_path = self.data_dir / "reviews.csv"

            if not data_path.exists():
                return Err("No text data found. Please upload a text file first.")

            data = pd.read_csv(data_path)

            if "content" not in data.columns:
                return Err("Missing 'content' column in data")

            # Generate labels from score if not present
            if "sentiment" not in data.columns:
                if "score" in data.columns and data["score"].notna().any():
                    data["sentiment"] = data["score"].apply(self._score_to_sentiment)
                else:
                    return Err("No sentiment labels found. Please include 'sentiment' or 'score' column.")

            # Drop rows with missing sentiment
            data = data.dropna(subset=["sentiment"])

            X = data[["content"]].copy()
            y = data["sentiment"].copy()

            return Ok((X, y))

        except Exception as e:
            return Err(f"Failed to load text dataset: {str(e)}")

    def preprocess_dataset(
        self,
        df: pd.DataFrame | None = None,
        preprocessor=None,
    ) -> Result[pd.DataFrame, str]:
        """Preprocess text dataset.

        Args:
            df: DataFrame with 'content' column. If None, loads from file.
            preprocessor: TextPreprocessor instance. If None, creates default.

        Returns:
            Ok(DataFrame with preprocessed text) if successful, Err with message if failed.
        """
        try:
            if preprocessor is None:
                from app.utils.text_preprocessing import TextPreprocessor
                preprocessor = TextPreprocessor()

            if df is None:
                data_path = self.data_dir / "reviews.csv"
                if not data_path.exists():
                    return Err("No data file found. Please upload a file first.")
                df = pd.read_csv(data_path)

            if "content" not in df.columns:
                return Err("Missing 'content' column")

            # Preprocess all texts
            df["preprocessed"] = df["content"].apply(preprocessor.preprocess)

            # Remove empty preprocessed texts
            df = df[df["preprocessed"].str.strip() != ""]

            # Save preprocessed data
            processed_path = self.data_dir / "reviews_preprocessed.csv"
            df.to_csv(processed_path, index=False, encoding="utf-8")

            return Ok(df)

        except Exception as e:
            return Err(f"Failed to preprocess dataset: {str(e)}")

    def get_preprocessed_texts(self) -> Result[tuple[list[str], list[str]], str]:
        """Get preprocessed texts and labels for training.

        Returns:
            Ok((texts, labels)) if successful, Err with message if failed.
        """
        try:
            data_path = self.data_dir / "reviews_preprocessed.csv"

            if not data_path.exists():
                # Try to preprocess first
                preprocess_result = self.preprocess_dataset()
                if preprocess_result.is_err():
                    return Err("No preprocessed data found. Please upload and preprocess data first.")
                df = preprocess_result.value
            else:
                df = pd.read_csv(data_path)

            # Filter out rows with missing preprocessed text
            df = df[df["preprocessed"].notna() & (df["preprocessed"] != "")]

            # Generate labels if needed
            if "sentiment" not in df.columns:
                if "score" in df.columns:
                    df["sentiment"] = df["score"].apply(self._score_to_sentiment)
                else:
                    return Err("No sentiment labels found")

            texts = df["preprocessed"].tolist()
            labels = df["sentiment"].tolist()

            return Ok((texts, labels))

        except Exception as e:
            return Err(f"Failed to get preprocessed texts: {str(e)}")

    def _is_valid_text_file(self, filename: str) -> bool:
        """Check if file has valid extension for text data.

        Args:
            filename: Name of the file.

        Returns:
            True if valid, False otherwise.
        """
        return any(filename.endswith(ext) for ext in [".csv", ".xls", ".xlsx"])

    @staticmethod
    def _score_to_sentiment(score: float | int | str | None) -> str | None:
        """Convert review score to sentiment label.

        Args:
            score: Review score (1-5).

        Returns:
            "Positif" for score 4-5, "Negatif" for score 1-2, None for score 3 or missing.
        """
        try:
            if pd.isna(score):
                return None
            score_val = int(float(score))
            if score_val >= 4:
                return "Positif"
            elif score_val <= 2:
                return "Negatif"
            else:
                return None  # Neutral (score 3) - excluded for binary classification
        except (ValueError, TypeError):
            return None

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
