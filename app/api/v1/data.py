"""Data processing API endpoints.

Handles file upload, data conversion, and reading operations.
"""

from io import BytesIO

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

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


@router.get(
    "/sample",
    summary="Download sample Excel file",
    description="Download a sample Excel file with the correct format for training data",
)
async def download_sample() -> StreamingResponse:
    """Download sample Excel file for sentiment analysis training data.

    Returns:
        StreamingResponse with Excel file.
    """
    # Create sample data for sentiment analysis
    sample_data = pd.DataFrame({
        "content": [
            "aplikasi sangat bagus dan membantu sekali",
            "selalu error tidak bisa dibuka bos",
            "mantap aplikasinya sangat berguna",
            "jelek banget selalu crash terus",
            "lumayan sih tapi masih banyak bug",
            "sangat puas dengan pelayanannya",
            "tolong diperbaiki lagi sering error",
        ],
        "score": [5, 1, 5, 1, 3, 5, 2],
    })

    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        sample_data.to_excel(writer, index=False, sheet_name="Data Training")

        # Add format info sheet
        format_info = pd.DataFrame({
            "Kolom": ["content", "score"],
            "Tipe Data": ["Text (String)", "Angka (1-5)"],
            "Wajib": ["Ya", "Tidak (opsional)"],
            "Keterangan": [
                "Isi review/ulasan aplikasi",
                "Rating 1-5 (4-5=Positif, 1-2=Negatif, 3=Netral/abaikan)"
            ],
        })
        format_info.to_excel(writer, index=False, sheet_name="Format")

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": "attachment; filename=sample_data_training.xlsx"
        },
    )
