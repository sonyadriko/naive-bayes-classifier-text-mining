"""Model evaluation API endpoints.

Handles model evaluation and metrics.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.schemas.prediction import EvaluationRequest
from app.services.model_service import ModelService
from app.utils.response import ApiResponse

router = APIRouter()


def get_model_service() -> ModelService:
    """Get model service instance."""
    return ModelService()


ModelService = Annotated[ModelService, Depends(get_model_service)]


@router.post(
    "/confusion-matrix",
    summary="Evaluate model with confusion matrix",
    description="Evaluate model using train-test split and generate metrics",
)
async def evaluate_confusion_matrix(
    request: EvaluationRequest,
    model_service: ModelService,
) -> JSONResponse:
    """Evaluate model with confusion matrix.

    Args:
        request: Evaluation request with test_size.
        model_service: Model service instance.

    Returns:
        JSONResponse with evaluation metrics.
    """
    result = model_service.evaluate(test_size=request.test_size)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Evaluation completed successfully",
        ),
    )
