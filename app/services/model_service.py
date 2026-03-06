"""Model service for training and predictions.

Handles model training, caching, and predictions.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.models.naive_bayes import ModelCache, NaiveBayesClassifier, PredictionResult
from app.services.data_service import DataService
from app.utils.response import Err, Ok, Result

settings = get_settings()


class ModelService:
    """Service for model training and predictions.

    Handles training the Naive Bayes classifier, caching the model,
    and making predictions.
    """

    def __init__(self) -> None:
        """Initialize model service."""
        self.data_service = DataService()
        self.model_cache = ModelCache()
        self._classifier: NaiveBayesClassifier | None = None

    def get_or_train_model(self) -> Result[NaiveBayesClassifier, str]:
        """Get cached model or train a new one.

        Returns:
            Ok(classifier) if successful, Err with message if failed.
        """
        # Try to load from cache
        cache_result = self.model_cache.load("current")
        if cache_result.is_ok() and cache_result.value is not None:
            self._classifier = cache_result.value
            return Ok(self._classifier)

        # Train new model
        return self._train_model()

    def _train_model(self) -> Result[NaiveBayesClassifier, str]:
        """Train a new model.

        Returns:
            Ok(classifier) if successful, Err with message if failed.
        """
        # Get training data
        data_result = self.data_service.get_training_data()
        if data_result.is_err():
            return Err(f"Failed to get training data: {data_result.error}")

        X, y = data_result.value

        # Create and train classifier
        classifier = NaiveBayesClassifier()
        train_result = classifier.train(X, y)

        if train_result.is_err():
            return Err(f"Training failed: {train_result.error}")

        # Cache the model
        self.model_cache.save(classifier, "current")
        self._classifier = classifier

        return Ok(classifier)

    def predict(self, features: dict[str, Any]) -> Result[PredictionResult, str]:
        """Make a prediction.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Ok(PredictionResult) if successful, Err with message if failed.
        """
        # Ensure model is trained
        model_result = self.get_or_train_model()
        if model_result.is_err():
            return Err(f"Failed to get model: {model_result.error}")

        classifier = model_result.value

        # Make prediction
        prediction_result = classifier.predict(features)

        if prediction_result.is_err():
            return Err(f"Prediction failed: {prediction_result.error}")

        return Ok(prediction_result.value)

    def evaluate(
        self,
        test_size: float = 0.2,
    ) -> Result[dict[str, Any], str]:
        """Evaluate model using train-test split.

        Args:
            test_size: Proportion of data for testing.

        Returns:
            Ok(evaluation metrics) if successful, Err with message if failed.
        """
        try:
            from sklearn.metrics import (
                accuracy_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import GaussianNB

            # Get training data
            data_result = self.data_service.get_training_data()
            if data_result.is_err():
                return Err(f"Failed to get training data: {data_result.error}")

            X, y = data_result.value

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train model
            nb = GaussianNB()
            nb.fit(X_train, y_train)

            # Make predictions
            y_pred = nb.predict(X_test)

            # Calculate metrics
            cm = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)

            # Handle binary vs multiclass metrics
            avg_method = "binary" if len(nb.classes_) == 2 else "weighted"

            precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)

            return Ok(
                {
                    "test_size": test_size,
                    "confusion_matrix": cm.tolist(),
                    "accuracy": round(accuracy, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1, 4),
                }
            )

        except Exception as e:
            return Err(f"Evaluation failed: {str(e)}")

    def clear_cache(self) -> Result[None, str]:
        """Clear cached model.

        Returns:
            Ok(None) if successful, Err with message if failed.
        """
        result = self.model_cache.clear("current")
        self._classifier = None
        return result

    def get_model_info(self) -> Result[dict[str, Any], str]:
        """Get information about the current model.

        Returns:
            Ok(model info) if successful, Err with message if failed.
        """
        if self._classifier is None:
            cache_result = self.model_cache.load("current")
            if cache_result.is_ok() and cache_result.value is not None:
                self._classifier = cache_result.value
            else:
                return Err("No trained model available")

        return Ok(
            {
                "is_trained": self._classifier.is_trained(),
                "classes": self._classifier.get_classes(),
                "feature_columns": self._classifier.get_feature_columns(),
            }
        )
