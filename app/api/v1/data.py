"""Data processing API endpoints.

Handles file upload, data conversion, and reading operations.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.data import DataConvertRequest
from app.services.data_service import DataService
from app.utils.response import ApiResponse

router = APIRouter()


def get_data_service() -> DataService:
    """Get data service instance."""
    return DataService()


DataService = Annotated[DataService, Depends(get_data_service)]


@router.post(
    "/upload",
    summary="Upload training data",
    description="Upload an Excel file with training data for the model",
)
async def upload_file(
    data_service: DataService,
    file: UploadFile = File(..., description="Excel file with training data"),
) -> JSONResponse:
    """Upload and process Excel file.

    Args:
        file: Uploaded Excel file.
        data_service: Data service instance.

    Returns:
        JSONResponse with processing result.
    """
    # Read file content
    content = await file.read()

    # Process file
    result = data_service.process_upload(content, file.filename)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="File processed successfully",
        ),
    )


@router.post(
    "/convert",
    summary="Convert categorical data",
    description="Convert categorical feature values to encoded values",
)
async def convert_data(
    request: DataConvertRequest,
    data_service: DataService,
) -> JSONResponse:
    """Convert categorical data.

    Args:
        request: Conversion request with input data.
        data_service: Data service instance.

    Returns:
        JSONResponse with converted data.
    """
    result = data_service.convert_data(request.input_data)

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Data converted successfully",
        ),
    )


@router.get(
    "/read",
    summary="Read training data",
    description="Read the uploaded training data from Excel file",
)
async def read_data(
    data_service: DataService,
) -> JSONResponse:
    """Read data from Excel file.

    Args:
        data_service: Data service instance.

    Returns:
        JSONResponse with data rows.
    """
    result = data_service.read_data()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"rows": result.value, "total": len(result.value)},
            message="Data read successfully",
        ),
    )


@router.get(
    "/labels",
    summary="Get label mappings",
    description="Get label encoder mappings for categorical features",
)
async def get_labels(
    data_service: DataService,
) -> JSONResponse:
    """Get label encoder mappings.

    Args:
        data_service: Data service instance.

    Returns:
        JSONResponse with label mappings.
    """
    result = data_service.get_labels()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"labels": result.value},
            message="Labels retrieved successfully",
        ),
    )


@router.get(
    "/info",
    summary="Get dataset information",
    description="Get information about the uploaded dataset",
)
async def get_data_info(
    data_service: DataService,
) -> JSONResponse:
    """Get dataset information.

    Args:
        data_service: Data service instance.

    Returns:
        JSONResponse with dataset info.
    """
    result = data_service.get_data_info()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Dataset info retrieved successfully",
        ),
    )
