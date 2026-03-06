"""Pydantic schemas for request/response validation."""

from app.schemas.auth import (
    LoginResponse,
    RefreshTokenRequest,
    RegisterRequest,
    TokenPayload,
    TokenResponse,
    UserInfoInLogin,
    UserInToken,
)
from app.schemas.data import (
    DataConvertRequest,
    DataConvertResponse,
    DataInfoResponse,
    DataReadResponse,
    DataRow,
    DataUploadResponse,
    FeatureInfo,
    LabelInfo,
    LabelsResponse,
)
from app.schemas.prediction import (
    BatchSentimentRequest,
    CategoricalPredictionRequest,
    ConfusionMatrixResponse,
    EvaluationRequest,
    LikelihoodDetail,
    PredictionResponse,
    SentimentPredictionRequest,
    SentimentPredictionResponse,
    TrainModelRequest,
)
from app.schemas.scraper import (
    ScrapeRequest,
    ScrapeResponse,
    ScraperStatusResponse,
)
from app.schemas.user import (
    BaseUserSchema,
    UserChangePassword,
    UserCreate,
    UserListResponse,
    UserLogin,
    UserResponse,
    UserUpdate,
)

__all__ = [
    # Auth schemas
    "TokenPayload",
    "TokenResponse",
    "UserInToken",
    "LoginResponse",
    "UserInfoInLogin",
    "RegisterRequest",
    "RefreshTokenRequest",
    # User schemas
    "BaseUserSchema",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserListResponse",
    "UserLogin",
    "UserChangePassword",
    # Prediction schemas
    "CategoricalPredictionRequest",
    "PredictionResponse",
    "LikelihoodDetail",
    "SentimentPredictionRequest",
    "SentimentPredictionResponse",
    "BatchSentimentRequest",
    "EvaluationRequest",
    "ConfusionMatrixResponse",
    "TrainModelRequest",
    # Data schemas
    "LabelInfo",
    "LabelsResponse",
    "DataUploadResponse",
    "DataConvertRequest",
    "DataConvertResponse",
    "DataRow",
    "DataReadResponse",
    "FeatureInfo",
    "DataInfoResponse",
    # Scraper schemas
    "ScrapeRequest",
    "ScrapeResponse",
    "ScraperStatusResponse",
]
