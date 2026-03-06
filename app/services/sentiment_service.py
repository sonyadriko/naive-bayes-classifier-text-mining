"""Sentiment analysis service.

Handles training, prediction, and model management for sentiment analysis.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.models.naive_bayes import ModelCache, MultinomialNaiveBayes, TextPredictionResult
from app.utils.response import Err, Ok, Result
from app.utils.text_preprocessing import TextPreprocessor

settings = get_settings()


class SentimentService:
    """Service for sentiment analysis operations.

    Handles model training, caching, and predictions for text sentiment analysis.
    """

    def __init__(self) -> None:
        """Initialize sentiment service."""
        from app.services.data_service import DataService

        self.data_service = DataService()
        self.preprocessor = TextPreprocessor()
        self.model_cache = ModelCache()
        self._classifier: MultinomialNaiveBayes | None = None
        self._vocabulary: list[str] | None = None

    def get_or_train_model(self, force_retrain: bool = False) -> Result[MultinomialNaiveBayes, str]:
        """Get cached model or train a new one.

        Args:
            force_retrain: Force retraining even if cached model exists.

        Returns:
            Ok(classifier) if successful, Err with message if failed.
        """
        # Try to load from cache (unless forced retrain)
        if not force_retrain:
            cache_result = self.model_cache.load("sentiment")
            if cache_result.is_ok() and cache_result.value is not None:
                self._classifier = cache_result.value
                return Ok(self._classifier)

        # Train new model
        return self._train_model()

    def _train_model(self) -> Result[MultinomialNaiveBayes, str]:
        """Train a new sentiment model.

        Returns:
            Ok(classifier) if successful, Err with message if failed.
        """
        # Get preprocessed training data
        data_result = self.data_service.get_preprocessed_texts()
        if data_result.is_err():
            return Err(f"Failed to get training data: {data_result.error}")

        texts, labels = data_result.value

        if not texts or not labels:
            return Err("No training data available. Please upload and preprocess data first.")

        # Build vocabulary from texts
        vocab = self._build_vocabulary_from_texts(texts)

        # Create and train classifier
        classifier = MultinomialNaiveBayes(alpha=1.0)
        train_result = classifier.train(texts, labels, vocab)

        if train_result.is_err():
            return Err(f"Training failed: {train_result.error}")

        # Cache the model
        self.model_cache.save(classifier, "sentiment")
        self._classifier = classifier
        self._vocabulary = vocab

        return Ok(classifier)

    def predict(self, text: str) -> Result[TextPredictionResult, str]:
        """Make a sentiment prediction.

        Args:
            text: Input text to analyze.

        Returns:
            Ok(TextPredictionResult) if successful, Err with message if failed.
        """
        # Ensure model is trained
        model_result = self.get_or_train_model()
        if model_result.is_err():
            return Err(f"Failed to get model: {model_result.error}")

        classifier = model_result.value

        # Preprocess text
        preprocessed = self.preprocessor.preprocess(text)

        if not preprocessed:
            # Return neutral result for empty/no-content text
            return Ok(
                TextPredictionResult(
                    text=text,
                    predicted_class="Netral",
                    confidence=0.5,
                    posteriors={"Positif": 0.5, "Negatif": 0.5},
                    priors={"Positif": 0.5, "Negatif": 0.5},
                )
            )

        # Make prediction
        prediction_result = classifier.predict(preprocessed)

        if prediction_result.is_err():
            return Err(f"Prediction failed: {prediction_result.error}")

        return Ok(prediction_result.value)

    def predict_batch(self, texts: list[str]) -> Result[list[TextPredictionResult], str]:
        """Make predictions for multiple texts.

        Args:
            texts: List of input texts.

        Returns:
            Ok(list of TextPredictionResult) if successful, Err with message if failed.
        """
        # Ensure model is trained
        model_result = self.get_or_train_model()
        if model_result.is_err():
            return Err(f"Failed to get model: {model_result.error}")

        classifier = model_result.value

        # Preprocess all texts
        preprocessed_texts = [self.preprocessor.preprocess(text) for text in texts]

        # Make predictions
        prediction_result = classifier.predict_batch(preprocessed_texts)

        if prediction_result.is_err():
            return Err(f"Batch prediction failed: {prediction_result.error}")

        # Update original text in results
        results = prediction_result.value
        for i, result in enumerate(results):
            result.text = texts[i]

        return Ok(results)

    def train_model(self, force_retrain: bool = False) -> Result[dict[str, Any], str]:
        """Train or retrain the sentiment model.

        Args:
            force_retrain: Force retraining even if cached model exists.

        Returns:
            Ok(dict with training info) if successful, Err with message if failed.
        """
        # Get or train model
        model_result = self.get_or_train_model(force_retrain=force_retrain)
        if model_result.is_err():
            return Err(model_result.error)

        classifier = model_result.value

        # Get training stats
        data_result = self.data_service.get_preprocessed_texts()
        if data_result.is_ok():
            texts, labels = data_result.value
            from collections import Counter

            label_counts = Counter(labels)

            return Ok(
                {
                    "model_type": "MultinomialNaiveBayes",
                    "vocabulary_size": len(classifier.get_vocabulary()),
                    "classes": classifier.get_classes(),
                    "training_samples": len(texts),
                    "label_distribution": dict(label_counts),
                    "is_trained": classifier.is_trained(),
                }
            )

        return Ok(
            {
                "model_type": "MultinomialNaiveBayes",
                "vocabulary_size": len(classifier.get_vocabulary()),
                "classes": classifier.get_classes(),
                "is_trained": classifier.is_trained(),
            }
        )

    def get_model_info(self) -> Result[dict[str, Any], str]:
        """Get information about the current model.

        Returns:
            Ok(model info) if successful, Err with message if failed.
        """
        if self._classifier is None:
            cache_result = self.model_cache.load("sentiment")
            if cache_result.is_ok() and cache_result.value is not None:
                self._classifier = cache_result.value
            else:
                return Err("No trained model available")

        return Ok(
            {
                "model_type": "MultinomialNaiveBayes",
                "is_trained": self._classifier.is_trained(),
                "classes": self._classifier.get_classes(),
                "vocabulary_size": len(self._classifier.get_vocabulary()),
            }
        )

    def clear_cache(self) -> Result[None, str]:
        """Clear cached model.

        Returns:
            Ok(None) if successful, Err with message if failed.
        """
        result = self.model_cache.clear("sentiment")
        self._classifier = None
        self._vocabulary = None
        return result

    def _build_vocabulary_from_texts(self, texts: list[str]) -> list[str]:
        """Build vocabulary from preprocessed texts.

        Args:
            texts: List of preprocessed texts.

        Returns:
            List of unique words (vocabulary).
        """
        vocab = set()
        for text in texts:
            if text:
                vocab.update(text.split())

        # Filter by minimum frequency (optional)
        from collections import Counter

        word_freq = Counter()
        for text in texts:
            if text:
                words = text.split()
                word_freq.update(words)

        # Keep words that appear at least twice
        min_freq = 2
        filtered_vocab = [word for word, count in word_freq.items() if count >= min_freq]

        return filtered_vocab if filtered_vocab else list(vocab)
