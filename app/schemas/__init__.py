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
    ConfusionMatrixResponse,
    EvaluationRequest,
    LikelihoodDetail,
    PredictionRequest,
    PredictionResponse,
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
    "PredictionRequest",
    "PredictionResponse",
    "LikelihoodDetail",
    "EvaluationRequest",
    "ConfusionMatrixResponse",
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
]
