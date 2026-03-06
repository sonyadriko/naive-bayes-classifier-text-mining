"""Refactored Naive Bayes Classifier with clean code principles.

This module implements a clean, DRY, and testable Naive Bayes classifier
following single responsibility principle and using Result type for error handling.

Includes:
- NaiveBayesClassifier: Gaussian/categorical Naive Bayes for general features
- MultinomialNaiveBayes: Multinomial Naive Bayes for text classification
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.utils.response import Err, Ok, Result

# Configuration constants
SMOOTHING_ALPHA = 1e-6  # Small probability for unseen values
LAPLACE_SMOOTHING = 1.0  # Laplace smoothing for MultinomialNB


@dataclass
class PredictionResult:
    """Result of a prediction with all relevant information.

    Attributes:
        predicted_class: The predicted class label.
        posteriors: Normalized posterior probabilities for each class.
        priors: Prior probabilities for each class.
        likelihoods: Likelihood values for each feature of the input.
        evidence: The evidence (normalization factor) for the prediction.
    """

    predicted_class: Any
    posteriors: dict[str | int, float]
    priors: dict[str | int, float]
    likelihoods: dict[str | int, dict[str, float]]
    evidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction result to dictionary.

        Returns:
            Dictionary representation of prediction result.
        """
        return {
            "predicted_class": self._convert_to_serializable(self.predicted_class),
            "posteriors": self._convert_dict(self.posteriors),
            "priors": self._convert_dict(self.priors),
            "likelihoods": self._convert_nested_dict(self.likelihoods),
            "evidence": float(self.evidence),
        }

    @staticmethod
    def _convert_to_serializable(value: Any) -> Any:
        """Convert numpy types to native Python types."""
        if isinstance(value, (np.integer, np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.floating, np.float64, np.float32)):
            return round(float(value), 4)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    @staticmethod
    def _convert_dict(d: dict) -> dict:
        """Convert dictionary values to serializable types."""
        return {
            PredictionResult._convert_to_serializable(k): PredictionResult._convert_to_serializable(v)
            for k, v in d.items()
        }

    @staticmethod
    def _convert_nested_dict(d: dict) -> dict:
        """Convert nested dictionary values to serializable types."""
        return {
            PredictionResult._convert_to_serializable(k): {
                PredictionResult._convert_to_serializable(k2): PredictionResult._convert_to_serializable(v2)
                for k2, v2 in v.items()
            }
            for k, v in d.items()
        }


class NaiveBayesClassifier:
    """Clean Naive Bayes Classifier implementation.

    Features:
    - Single responsibility: Each method does one thing
    - DRY: Common operations extracted to private methods
    - Result type: Error handling without exceptions
    - Model caching: Train once, use many times
    - Type hints: Full type annotations

    Example:
        ```python
        classifier = NaiveBayesClassifier()
        result = classifier.train(X, y)
        if result.is_err():
            print(result.error)

        prediction = classifier.predict({"feature1": "value1"})
        if prediction.is_ok():
            print(prediction.value.predicted_class)
        ```
    """

    def __init__(self) -> None:
        """Initialize the classifier."""
        self._priors: dict[Any, float] = {}
        self._likelihoods: dict[Any, dict[str, dict[Any, float]]] = {}
        self._classes: list[Any] = []
        self._feature_columns: list[str] = []
        self._is_trained_flag: bool = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> Result[None, str]:
        """Train the Naive Bayes model.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Ok(None) if training succeeds, Err with message if it fails.
        """
        try:
            # Validate inputs
            validation_result = self._validate_training_data(X, y)
            if validation_result.is_err():
                return validation_result

            # Store feature columns
            self._feature_columns = X.columns.tolist()

            # Extract unique classes
            self._classes = self._get_unique_classes(y)

            # Calculate priors P(C)
            self._priors = self._calculate_priors(y)

            # Calculate likelihoods P(X|C)
            self._likelihoods = self._calculate_likelihoods(X, y)

            # Mark as trained
            self._is_trained_flag = True

            return Ok(None)

        except Exception as e:
            return Err(f"Training failed: {str(e)}")

    def predict(self, features: dict[str, Any]) -> Result[PredictionResult, str]:
        """Make a prediction for a single instance.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Ok(PredictionResult) if prediction succeeds, Err with message if it fails.
        """
        # Check if model is trained
        if not self._is_trained_flag:
            return Err("Model must be trained before making predictions")

        try:
            # Validate features
            validation_result = self._validate_features(features)
            if validation_result.is_err():
                return validation_result

            # Calculate posteriors for each class
            posteriors = self._calculate_posteriors(features)

            # Get predicted class (max posterior)
            predicted_class = self._get_max_class(posteriors)

            # Calculate evidence
            evidence = sum(posteriors.values())

            # Normalize posteriors
            normalized_posteriors = self._normalize_posteriors(posteriors, evidence)

            # Extract feature likelihoods for the input
            feature_likelihoods = self._get_feature_likelihoods(features)

            return Ok(
                PredictionResult(
                    predicted_class=predicted_class,
                    posteriors=normalized_posteriors,
                    priors=self._priors.copy(),
                    likelihoods=feature_likelihoods,
                    evidence=evidence,
                )
            )

        except Exception as e:
            return Err(f"Prediction failed: {str(e)}")

    def is_trained(self) -> bool:
        """Check if the model has been trained.

        Returns:
            True if model is trained, False otherwise.
        """
        return self._is_trained_flag

    def get_classes(self) -> list[Any]:
        """Get the list of classes.

        Returns:
            List of unique class labels.
        """
        return self._classes.copy()

    def get_feature_columns(self) -> list[str]:
        """Get the feature column names.

        Returns:
            List of feature column names.
        """
        return self._feature_columns.copy()

    # Private helper methods - DRY principle

    def _validate_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Result[None, str]:
        """Validate training data.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Ok(None) if valid, Err with message if invalid.
        """
        if len(X) == 0:
            return Err("Training data cannot be empty")

        if len(X) != len(y):
            return Err("Features and target must have the same length")

        if X.columns.empty:
            return Err("Features must have at least one column")

        return Ok(None)

    def _validate_features(self, features: dict[str, Any]) -> Result[None, str]:
        """Validate prediction features.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Ok(None) if valid, Err with message if invalid.
        """
        if not features:
            return Err("Features cannot be empty")

        missing_features = set(self._feature_columns) - set(features.keys())
        if missing_features:
            return Err(f"Missing required features: {missing_features}")

        return Ok(None)

    def _get_unique_classes(self, y: pd.Series) -> list[Any]:
        """Get unique class labels from target.

        Args:
            y: Target Series.

        Returns:
            List of unique class labels.
        """
        return list(y.unique())

    def _calculate_priors(self, y: pd.Series) -> dict[Any, float]:
        """Calculate prior probabilities P(C) for each class.

        Args:
            y: Target Series.

        Returns:
            Dictionary mapping class to prior probability.
        """
        total_samples = len(y)
        priors = {}

        for cls in self._classes:
            class_count = (y == cls).sum()
            priors[cls] = class_count / total_samples

        return priors

    def _calculate_likelihoods(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[Any, dict[str, dict[Any, float]]]:
        """Calculate likelihood probabilities P(X|C) for each feature.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Nested dictionary of likelihoods: class -> feature -> value -> probability.
        """
        likelihoods = {}

        for cls in self._classes:
            # Get data for this class
            class_data = X[y == cls]
            class_likelihoods = {}

            for column in X.columns:
                feature_likelihoods = {}
                value_counts = class_data[column].value_counts()

                # Calculate P(feature_value|class) with smoothing
                class_count = len(class_data)
                for value, count in value_counts.items():
                    probability = (count + SMOOTHING_ALPHA) / (class_count + SMOOTHING_ALPHA * 2)
                    feature_likelihoods[value] = round(probability, 6)

                class_likelihoods[column] = feature_likelihoods

            likelihoods[cls] = class_likelihoods

        return likelihoods

    def _calculate_posteriors(self, features: dict[str, Any]) -> dict[Any, float]:
        """Calculate unnormalized posterior probabilities for each class.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Dictionary mapping class to unnormalized posterior probability.
        """
        posteriors = {}

        for cls in self._classes:
            prior = self._priors.get(cls, 0)
            likelihood = self._calculate_class_likelihood(cls, features)
            posteriors[cls] = prior * likelihood

        return posteriors

    def _calculate_class_likelihood(self, cls: Any, features: dict[str, Any]) -> float:
        """Calculate likelihood P(X|C) for a given class.

        Args:
            cls: Class label.
            features: Dictionary of feature names to values.

        Returns:
            Likelihood probability.
        """
        likelihood = 1.0

        for feature, value in features.items():
            feature_likelihoods = self._likelihoods.get(cls, {}).get(feature, {})

            if value in feature_likelihoods:
                likelihood_value = feature_likelihoods[value]
            else:
                # Use small probability for unseen values
                likelihood_value = SMOOTHING_ALPHA

            likelihood *= likelihood_value

        return likelihood

    def _normalize_posteriors(
        self,
        posteriors: dict[Any, float],
        evidence: float,
    ) -> dict[Any, float]:
        """Normalize posterior probabilities.

        Args:
            posteriors: Unnormalized posterior probabilities.
            evidence: Normalization factor (sum of posteriors).

        Returns:
            Dictionary mapping class to normalized posterior probability.
        """
        if evidence == 0:
            # Return equal probabilities if evidence is zero
            return {cls: 1.0 / len(self._classes) for cls in self._classes}

        return {
            cls: round(posterior / evidence, 6)
            for cls, posterior in posteriors.items()
        }

    def _get_max_class(self, posteriors: dict[Any, float]) -> Any:
        """Get the class with maximum posterior probability.

        Args:
            posteriors: Dictionary mapping class to posterior probability.

        Returns:
            Class label with maximum probability.
        """
        return max(posteriors, key=posteriors.get)

    def _get_feature_likelihoods(
        self,
        features: dict[str, Any],
    ) -> dict[Any, dict[str, float]]:
        """Get likelihood values for each feature of the input.

        Args:
            features: Dictionary of feature names to values.

        Returns:
            Dictionary mapping class to feature likelihoods.
        """
        result = {}

        for cls in self._classes:
            class_likelihoods = {}

            for feature, value in features.items():
                feature_likelihoods = self._likelihoods.get(cls, {}).get(feature, {})

                if value in feature_likelihoods:
                    class_likelihoods[feature] = feature_likelihoods[value]
                else:
                    class_likelihoods[feature] = SMOOTHING_ALPHA

            result[cls] = class_likelihoods

        return result


class MultinomialNaiveBayes:
    """Multinomial Naive Bayes classifier for text classification.

    This classifier is designed for text data with Bag of Words features.
    It uses multinomial distribution with Laplace smoothing for better
    performance on text data.

    Features:
    - Optimized for text/Bag of Words features
    - Laplace smoothing for unseen words
    - Efficient likelihood calculation for sparse features
    - Works with TextPreprocessor and BagOfWords classes

    Example:
        ```python
        from app.utils.text_preprocessing import TextPreprocessor, BagOfWords

        preprocessor = TextPreprocessor()
        bow = BagOfWords()

        # Prepare training data
        texts = ["aplikasi bagus", "aplikasi jelek"]
        labels = ["Positif", "Negatif"]

        # Preprocess and create BOW
        preprocessed = preprocessor.preprocess_batch(texts)
        bow.fit(preprocessed)

        # Train classifier
        clf = MultinomialNaiveBayes()
        clf.train(preprocessed, labels, bow.get_vocabulary())

        # Predict
        new_text = preprocessor.preprocess("aplikasi sangat bagus")
        result = clf.predict(new_text)
        ```
    """

    def __init__(self, alpha: float = LAPLACE_SMOOTHING) -> None:
        """Initialize Multinomial Naive Bayes classifier.

        Args:
            alpha: Laplace smoothing parameter (default: 1.0).
        """
        self.alpha = alpha
        self._priors: dict[str, float] = {}
        self._log_likelihoods: dict[str, dict[str, float]] = {}
        self._classes: list[str] = []
        self._vocabulary: set[str] = set()
        self._class_word_counts: dict[str, int] = {}
        self._is_trained_flag: bool = False

    def train(
        self,
        texts: list[str],
        labels: list[str],
        vocabulary: list[str] | set[str] | None = None,
    ) -> Result[None, str]:
        """Train the Multinomial Naive Bayes model.

        Args:
            texts: List of preprocessed texts (space-separated tokens).
            labels: List of class labels for each text.
            vocabulary: Vocabulary to use (if None, extracted from texts).

        Returns:
            Ok(None) if training succeeds, Err with message if it fails.
        """
        try:
            # Validate inputs
            validation_result = self._validate_training_data(texts, labels)
            if validation_result.is_err():
                return validation_result

            # Build vocabulary if not provided
            if vocabulary is None:
                self._vocabulary = self._build_vocabulary(texts)
            else:
                self._vocabulary = set(vocabulary)

            if not self._vocabulary:
                return Err("Vocabulary is empty. Cannot train with empty vocabulary.")

            # Extract unique classes
            self._classes = list(set(labels))

            # Calculate priors P(Class)
            self._priors = self._calculate_priors(labels)

            # Calculate likelihoods P(word|class) with Laplace smoothing
            self._log_likelihoods = self._calculate_likelihoods(texts, labels)

            # Mark as trained
            self._is_trained_flag = True

            return Ok(None)

        except Exception as e:
            return Err(f"Training failed: {str(e)}")

    def predict(self, text: str) -> Result["TextPredictionResult", str]:
        """Make a prediction for a single text.

        Args:
            text: Preprocessed text (space-separated tokens).

        Returns:
            Ok(TextPredictionResult) if successful, Err with message if failed.
        """
        if not self._is_trained_flag:
            return Err("Model must be trained before making predictions")

        try:
            tokens = text.split() if text else []

            # Calculate log posteriors for each class
            log_posteriors = self._calculate_log_posteriors(tokens)

            # Get predicted class (max log posterior)
            predicted_class = self._get_max_class(log_posteriors)

            # Convert to probabilities
            posteriors = self._log_to_probs(log_posteriors)

            # Get confidence (max posterior probability)
            confidence = posteriors.get(predicted_class, 0.0)

            return Ok(
                TextPredictionResult(
                    text=text,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    posteriors=posteriors,
                    priors=self._priors.copy(),
                )
            )

        except Exception as e:
            return Err(f"Prediction failed: {str(e)}")

    def predict_batch(self, texts: list[str]) -> Result[list["TextPredictionResult"], str]:
        """Make predictions for multiple texts.

        Args:
            texts: List of preprocessed texts.

        Returns:
            Ok(list of TextPredictionResult) if successful, Err with message if failed.
        """
        if not self._is_trained_flag:
            return Err("Model must be trained before making predictions")

        try:
            results = []
            for text in texts:
                result = self.predict(text)
                if result.is_err():
                    return Err(f"Prediction failed for text '{text[:30]}...': {result.error}")
                results.append(result.value)

            return Ok(results)

        except Exception as e:
            return Err(f"Batch prediction failed: {str(e)}")

    def is_trained(self) -> bool:
        """Check if the model has been trained.

        Returns:
            True if model is trained, False otherwise.
        """
        return self._is_trained_flag

    def get_classes(self) -> list[str]:
        """Get the list of classes.

        Returns:
            List of unique class labels.
        """
        return self._classes.copy()

    def get_vocabulary(self) -> set[str]:
        """Get the vocabulary.

        Returns:
            Set of vocabulary words.
        """
        return self._vocabulary.copy()

    def get_feature_log_likelihoods(self, word: str, cls: str | None = None) -> dict[str, float] | float:
        """Get log likelihood for a word.

        Args:
            word: Word to get likelihood for.
            cls: Specific class, or None for all classes.

        Returns:
            Log likelihood P(word|class) for the word(s).
        """
        if cls:
            return self._log_likelihoods.get(cls, {}).get(word, -np.inf)
        return {c: probs.get(word, -np.inf) for c, probs in self._log_likelihoods.items()}

    # Private helper methods

    def _validate_training_data(
        self,
        texts: list[str],
        labels: list[str],
    ) -> Result[None, str]:
        """Validate training data.

        Args:
            texts: List of preprocessed texts.
            labels: List of class labels.

        Returns:
            Ok(None) if valid, Err with message if invalid.
        """
        if len(texts) == 0:
            return Err("Training data cannot be empty")

        if len(texts) != len(labels):
            return Err("Texts and labels must have the same length")

        return Ok(None)

    def _build_vocabulary(self, texts: list[str]) -> set[str]:
        """Build vocabulary from texts.

        Args:
            texts: List of preprocessed texts.

        Returns:
            Set of unique words.
        """
        vocab = set()
        for text in texts:
            if text:
                vocab.update(text.split())
        return vocab

    def _calculate_priors(self, labels: list[str]) -> dict[str, float]:
        """Calculate prior probabilities P(Class).

        Args:
            labels: List of class labels.

        Returns:
            Dictionary mapping class to prior probability.
        """
        total_samples = len(labels)
        priors = {}

        for cls in self._classes:
            class_count = labels.count(cls)
            priors[cls] = class_count / total_samples

        return priors

    def _calculate_likelihoods(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, dict[str, float]]:
        """Calculate log likelihood probabilities log(P(word|class)).

        Uses Laplace smoothing: P(w|c) = (count(w in c) + alpha) / (total_words_in_c + alpha * vocab_size)

        Args:
            texts: List of preprocessed texts.
            labels: List of class labels.

        Returns:
            Nested dictionary: class -> word -> log likelihood.
        """
        vocab_size = len(self._vocabulary)
        log_likelihoods = {}
        self._class_word_counts = {}

        for cls in self._classes:
            # Get all texts for this class
            class_texts = [t for t, l in zip(texts, labels) if l == cls]

            # Count word occurrences
            word_counts: dict[str, int] = {}
            total_words = 0

            for text in class_texts:
                if text:
                    words = text.split()
                    for word in words:
                        if word in self._vocabulary:
                            word_counts[word] = word_counts.get(word, 0) + 1
                            total_words += 1

            self._class_word_counts[cls] = total_words

            # Calculate log likelihoods with Laplace smoothing
            class_likelihoods = {}
            for word in self._vocabulary:
                count = word_counts.get(word, 0)
                # Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                likelihood = (count + self.alpha) / (total_words + self.alpha * vocab_size)
                class_likelihoods[word] = np.log(likelihood + 1e-10)  # Add small value to avoid log(0)

            log_likelihoods[cls] = class_likelihoods

        return log_likelihoods

    def _calculate_log_posteriors(self, tokens: list[str]) -> dict[str, float]:
        """Calculate log posterior probabilities for each class.

        log P(c|tokens) ∝ log P(c) + sum(log P(wi|c))

        Args:
            tokens: List of tokens from preprocessed text.

        Returns:
            Dictionary mapping class to log posterior probability.
        """
        log_posteriors = {}

        for cls in self._classes:
            # Start with log prior
            log_posterior = np.log(self._priors.get(cls, 1e-10))

            # Add log likelihoods for each token
            class_likelihoods = self._log_likelihoods.get(cls, {})

            for token in tokens:
                if token in self._vocabulary:
                    # Use log likelihood from training, or small value for unseen
                    log_likelihood = class_likelihoods.get(token, np.log(self.alpha / (
                        self._class_word_counts.get(cls, 1) + self.alpha * len(self._vocabulary)
                    )))
                    log_posterior += log_likelihood

            log_posteriors[cls] = log_posterior

        return log_posteriors

    def _log_to_probs(self, log_posteriors: dict[str, float]) -> dict[str, float]:
        """Convert log posteriors to probabilities using log-sum-exp trick.

        Args:
            log_posteriors: Dictionary mapping class to log posterior.

        Returns:
            Dictionary mapping class to normalized probability.
        """
        # Find max log value for numerical stability
        max_log = max(log_posteriors.values())

        # Subtract max and exponentiate
        exp_values = {cls: np.exp(val - max_log) for cls, val in log_posteriors.items()}

        # Normalize
        total = sum(exp_values.values())

        if total == 0:
            # Return equal probabilities if total is zero
            return {cls: 1.0 / len(self._classes) for cls in self._classes}

        return {cls: val / total for cls, val in exp_values.items()}

    def _get_max_class(self, log_posteriors: dict[str, float]) -> str:
        """Get the class with maximum log posterior.

        Args:
            log_posteriors: Dictionary mapping class to log posterior.

        Returns:
            Class label with maximum probability.
        """
        return max(log_posteriors, key=log_posteriors.get)

    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary for serialization.

        Returns:
            Dictionary representation of the model.
        """
        return {
            "priors": self._priors,
            "log_likelihoods": self._log_likelihoods,
            "classes": self._classes,
            "vocabulary": list(self._vocabulary),
            "class_word_counts": self._class_word_counts,
            "alpha": self.alpha,
            "is_trained": self._is_trained_flag,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultinomialNaiveBayes":
        """Create model from dictionary.

        Args:
            data: Dictionary representation of the model.

        Returns:
            MultinomialNaiveBayes instance.
        """
        model = cls(alpha=data.get("alpha", LAPLACE_SMOOTHING))
        model._priors = data.get("priors", {})
        model._log_likelihoods = data.get("log_likelihoods", {})
        model._classes = data.get("classes", [])
        model._vocabulary = set(data.get("vocabulary", []))
        model._class_word_counts = data.get("class_word_counts", {})
        model._is_trained_flag = data.get("is_trained", False)
        return model


@dataclass
class TextPredictionResult:
    """Result of a text prediction.

    Attributes:
        text: The preprocessed input text.
        predicted_class: The predicted class label.
        confidence: Confidence score (max posterior probability).
        posteriors: Normalized posterior probabilities for each class.
        priors: Prior probabilities for each class.
    """

    text: str
    predicted_class: str
    confidence: float
    posteriors: dict[str, float]
    priors: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of prediction result.
        """
        return {
            "text": self.text,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "posteriors": {k: round(v, 4) for k, v in self.posteriors.items()},
            "priors": {k: round(v, 4) for k, v in self.priors.items()},
        }


class ModelCache:
    """Cache for trained models to avoid retraining.

    Implements a simple file-based cache with TTL support.
    Supports both NaiveBayesClassifier and MultinomialNaiveBayes.
    """

    def __init__(self, cache_dir: str = "data/models") -> None:
        """Initialize model cache.

        Args:
            cache_dir: Directory to store cached models.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model: NaiveBayesClassifier | MultinomialNaiveBayes, key: str) -> Result[None, str]:
        """Save model to cache.

        Args:
            model: Trained model to cache.
            key: Cache key (typically file hash or identifier).

        Returns:
            Ok(None) if successful, Err with message if failed.
        """
        try:
            cache_path = self.cache_dir / f"{key}.pkl"
            with open(cache_path, "wb") as f:
                pickle.dump(model, f)
            return Ok(None)
        except Exception as e:
            return Err(f"Failed to save model to cache: {str(e)}")

    def load(self, key: str) -> Result[NaiveBayesClassifier | MultinomialNaiveBayes | None, str]:
        """Load model from cache.

        Args:
            key: Cache key.

        Returns:
            Ok(model) if found and valid, Ok(None) if not found, Err on failure.
        """
        try:
            cache_path = self.cache_dir / f"{key}.pkl"

            if not cache_path.exists():
                return Ok(None)

            with open(cache_path, "rb") as f:
                model = pickle.load(f)

            if isinstance(model, (NaiveBayesClassifier, MultinomialNaiveBayes)) and model.is_trained():
                return Ok(model)

            return Ok(None)

        except Exception as e:
            return Err(f"Failed to load model from cache: {str(e)}")

    def clear(self, key: str | None = None) -> Result[None, str]:
        """Clear cached models.

        Args:
            key: Specific key to clear, or None to clear all.

        Returns:
            Ok(None) if successful, Err with message if failed.
        """
        try:
            if key:
                cache_path = self.cache_dir / f"{key}.pkl"
                if cache_path.exists():
                    cache_path.unlink()
            else:
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()

            return Ok(None)

        except Exception as e:
            return Err(f"Failed to clear cache: {str(e)}")
