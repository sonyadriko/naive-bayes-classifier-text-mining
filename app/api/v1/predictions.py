"""Prediction API endpoints.

Handles model prediction requests.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.schemas.prediction import PredictionRequest
from app.services.model_service import ModelService
from app.utils.response import ApiResponse

router = APIRouter()


def get_model_service() -> ModelService:
    """Get model service instance."""
    return ModelService()


ModelService = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/predict",
    summary="Make a prediction",
    description="Predict job placement duration using Naive Bayes classifier",
)
async def predict(
    request: PredictionRequest,
    model_service: ModelService,
) -> JSONResponse:
    """Make prediction.

    Args:
        request: Prediction request with features.
        model_service: Model service instance.

    Returns:
        JSONResponse with prediction result.
    """
    # Convert request to dict
    features = request.model_dump()

    # Make prediction
    result = model_service.predict(features)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    # Convert prediction result to dict
    prediction_dict = result.value.to_dict()

    return JSONResponse(
        content=ApiResponse.success(
            data=prediction_dict,
            message="Prediction successful",
        ),
    )


@router.get(
    "/model/info",
    summary="Get model information",
    description="Get information about the current trained model",
)
async def get_model_info(
    model_service: ModelService,
) -> JSONResponse:
    """Get model information.

    Args:
        model_service: Model service instance.

    Returns:
        JSONResponse with model info.
    """
    result = model_service.get_model_info()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Model info retrieved successfully",
        ),
    )


@router.post(
    "/model/cache/clear",
    summary="Clear model cache",
    description="Clear the cached model to force retraining",
)
async def clear_model_cache(
    model_service: ModelService,
) -> JSONResponse:
    """Clear model cache.

    Args:
        model_service: Model service instance.

    Returns:
        JSONResponse confirming cache cleared.
    """
    result = model_service.clear_cache()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"cached": False},
            message="Model cache cleared",
        ),
    )
