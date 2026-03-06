"""Label encoding service.

Handles label encoding for categorical features with caching support.
"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from app.utils.response import Err, Ok, Result


class LabelService:
    """Service for managing label encoders.

    Provides label encoding functionality with file-based caching
    to persist encoders across requests.
    """

    def __init__(self, cache_dir: str = "data/encoders") -> None:
        """Initialize label service.

        Args:
            cache_dir: Directory to store encoder caches.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoders: dict[str, LabelEncoder] = {}

    def fit_encoders(self, data: pd.DataFrame) -> Result[dict[str, LabelEncoder], str]:
        """Fit label encoders on categorical columns.

        Args:
            data: DataFrame to fit encoders on.

        Returns:
            Ok(dict of encoders) if successful, Err with message if failed.
        """
        try:
            self._encoders = {}

            for column in data.columns:
                if data[column].dtype == "object":
                    encoder = LabelEncoder()
                    data[column] = encoder.fit_transform(data[column])
                    self._encoders[column] = encoder

            return Ok(self._encoders)

        except Exception as e:
            return Err(f"Failed to fit encoders: {str(e)}")

    def transform(self, input_data: dict[str, Any]) -> Result[dict[str, Any], str]:
        """Transform input data using fitted encoders.

        Args:
            input_data: Dictionary of feature names to values.

        Returns:
            Ok(transformed_data) if successful, Err with message if failed.
        """
        try:
            transformed = {}

            for key, value in input_data.items():
                if key in self._encoders:
                    # Handle unseen values by mapping to a default value
                    encoder = self._encoders[key]
                    try:
                        transformed[key] = int(encoder.transform([value])[0])
                    except ValueError:
                        # Unseen value - use 0 as default
                        transformed[key] = 0
                else:
                    transformed[key] = value

            return Ok(transformed)

        except Exception as e:
            return Err(f"Failed to transform data: {str(e)}")

    def get_mappings(self) -> Result[dict[str, dict[str, Any]], str]:
        """Get label mappings for all encoders.

        Args:
            None

        Returns:
            Ok(dict of label mappings) if successful, Err with message if failed.
        """
        if not self._encoders:
            return Err("No encoders fitted. Call fit_encoders first.")

        mappings = {}

        for column, encoder in self._encoders.items():
            mappings[column] = {
                "encoded_values": {i: str(label) for i, label in enumerate(encoder.classes_)},
                "classes": [str(label) for label in encoder.classes_],
            }

        return Ok(mappings)

    def save_encoders(self, key: str = "default") -> Result[None, str]:
        """Save encoders to cache.

        Args:
            key: Cache key for the encoders.

        Returns:
            Ok(None) if successful, Err with message if failed.
        """
        try:
            cache_path = self.cache_dir / f"{key}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(self._encoders, f)
            return Ok(None)

        except Exception as e:
            return Err(f"Failed to save encoders: {str(e)}")

    def load_encoders(self, key: str = "default") -> Result[dict[str, LabelEncoder], str]:
        """Load encoders from cache.

        Args:
            key: Cache key for the encoders.

        Returns:
            Ok(dict of encoders) if successful, Err with message if failed.
        """
        try:
            cache_path = self.cache_dir / f"{key}.pkl"

            if not cache_path.exists():
                return Err("No cached encoders found")

            with open(cache_path, "rb") as f:
                self._encoders = pickle.load(f)

            return Ok(self._encoders)

        except Exception as e:
            return Err(f"Failed to load encoders: {str(e)}")

    def has_encoders(self) -> bool:
        """Check if encoders are loaded.

        Returns:
            True if encoders exist, False otherwise.
        """
        return len(self._encoders) > 0

    def clear(self) -> None:
        """Clear all encoders."""
        self._encoders = {}
