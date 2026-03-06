"""Data models."""

from app.models.naive_bayes import ModelCache, NaiveBayesClassifier, PredictionResult
from app.models.user import Base, User

__all__ = [
    "Base",
    "User",
    "NaiveBayesClassifier",
    "PredictionResult",
    "ModelCache",
]
