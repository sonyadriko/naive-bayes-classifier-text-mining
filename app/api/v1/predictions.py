"""Prediction API endpoints.

Handles model prediction requests for both categorical features and sentiment analysis.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.models.naive_bayes import MultinomialNaiveBayes, TextPredictionResult
from app.schemas.prediction import (
    BatchSentimentRequest,
    CategoricalPredictionRequest,
    SentimentPredictionRequest,
    TrainModelRequest,
)
from app.services.model_service import ModelService
from app.services.sentiment_service import SentimentService
from app.utils.text_preprocessing import TextPreprocessor
from app.utils.response import ApiResponse

router = APIRouter()


def get_model_service() -> ModelService:
    """Get model service instance."""
    return ModelService()


def get_sentiment_service() -> SentimentService:
    """Get sentiment service instance."""
    return SentimentService()


ModelService = Annotated[ModelService, Depends(get_model_service)]
SentimentService = Annotated[SentimentService, Depends(get_sentiment_service)]


# === Sentiment Analysis Endpoints ===

@router.post(
    "/sentiment",
    summary="Analyze text sentiment",
    description="Predict sentiment (Positif/Negatif) for Indonesian text",
)
async def predict_sentiment(
    request: SentimentPredictionRequest,
    sentiment_service: SentimentService,
) -> JSONResponse:
    """Predict sentiment for a single text.

    Args:
        request: Sentiment prediction request with text.
        sentiment_service: Sentiment service instance.

    Returns:
        JSONResponse with sentiment prediction result.
    """
    result = sentiment_service.predict(request.text)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="PREDICTION_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value.to_dict(),
            message="Sentiment prediction successful",
        ),
    )


@router.post(
    "/sentiment/batch",
    summary="Analyze multiple texts for sentiment",
    description="Predict sentiment for multiple texts at once",
)
async def predict_sentiment_batch(
    request: BatchSentimentRequest,
    sentiment_service: SentimentService,
) -> JSONResponse:
    """Predict sentiment for multiple texts.

    Args:
        request: Batch sentiment prediction request.
        sentiment_service: Sentiment service instance.

    Returns:
        JSONResponse with batch prediction results.
    """
    result = sentiment_service.predict_batch(request.texts)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="BATCH_PREDICTION_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"predictions": [r.to_dict() for r in result.value]},
            message=f"Successfully analyzed {len(result.value)} texts",
        ),
    )


@router.post(
    "/sentiment/train",
    summary="Train sentiment model",
    description="Train or retrain the sentiment analysis model with current data",
)
async def train_sentiment_model(
    request: TrainModelRequest,
    sentiment_service: SentimentService,
) -> JSONResponse:
    """Train the sentiment analysis model.

    Args:
        request: Training request parameters.
        sentiment_service: Sentiment service instance.

    Returns:
        JSONResponse with training result.
    """
    result = sentiment_service.train_model(force_retrain=request.force_retrain)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="TRAINING_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Model trained successfully",
        ),
    )


# === Legacy Categorical Prediction Endpoints ===

@router.post(
    "/predict",
    summary="Make a prediction (legacy)",
    description="Predict job placement duration using Naive Bayes classifier",
)
async def predict(
    request: CategoricalPredictionRequest,
    model_service: ModelService,
) -> JSONResponse:
    """Make prediction (legacy categorical prediction).

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
            content=ApiResponse.error(message=result.error, code="PREDICTION_ERROR"),
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
    sentiment_service: SentimentService,
) -> JSONResponse:
    """Get model information.

    Args:
        sentiment_service: Sentiment service instance.

    Returns:
        JSONResponse with model info.
    """
    result = sentiment_service.get_model_info()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="INFO_ERROR"),
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
    sentiment_service: SentimentService,
) -> JSONResponse:
    """Clear model cache.

    Args:
        sentiment_service: Sentiment service instance.

    Returns:
        JSONResponse confirming cache cleared.
    """
    result = sentiment_service.clear_cache()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="CACHE_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"cached": False},
            message="Model cache cleared",
        ),
    )


@router.get(
    "/preprocess/sample",
    summary="Test text preprocessing",
    description="Preprocess a sample text to see the transformation",
)
async def preprocess_sample(
    text: str,
) -> JSONResponse:
    """Preprocess a sample text.

    Args:
        text: Text to preprocess.

    Returns:
        JSONResponse with preprocessed text.
    """
    try:
        preprocessor = TextPreprocessor()
        preprocessed = preprocessor.preprocess(text)

        # Get individual steps
        steps = {
            "original": text,
            "case_folded": preprocessor.case_folding(text),
            "cleaned": preprocessor.clean_text(preprocessor.case_folding(text)),
            "tokenized": preprocessor.tokenize(preprocessor.clean_text(preprocessor.case_folding(text))),
            "final": preprocessed,
        }

        return JSONResponse(
            content=ApiResponse.success(
                data=steps,
                message="Text preprocessed successfully",
            ),
        )

    except Exception as e:
        return JSONResponse(
            content=ApiResponse.error(message=str(e), code="PREPROCESS_ERROR"),
            status_code=400,
        )
