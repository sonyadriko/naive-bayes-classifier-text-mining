"""API router aggregation.

Combines all API routers.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.v1 import auth, data, evaluation, predictions, users
from app.core.database import get_db

# Type aliases for dependency injection
DBSession = Annotated[Session, Depends(get_db)]

# Re-export commonly used types from users module
CurrentUser = users.CurrentUser

# Placeholder for CurrentActiveUser (can be implemented later)
CurrentActiveUser = CurrentUser

# Create API router
api_router = APIRouter()

# Include all routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(data.router, prefix="/data", tags=["Data"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
api_router.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])

__all__ = [
    "api_router",
    "DBSession",
    "CurrentUser",
    "CurrentActiveUser",
]
