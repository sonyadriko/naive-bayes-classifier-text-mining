"""Scraper API endpoints.

Handles Google Play Store review scraping operations.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from app.schemas.scraper import (
    ScrapeRequest,
    ScrapeResponse,
    ScraperStatusResponse,
)
from app.services.scraper_service import GooglePlayScraper
from app.utils.response import ApiResponse

router = APIRouter()


def get_scraper() -> GooglePlayScraper:
    """Get scraper instance."""
    return GooglePlayScraper()


Scraper = Annotated[GooglePlayScraper, Depends(get_scraper)]


@router.post(
    "/reviews",
    summary="Scrape Google Play Store reviews",
    description="Fetch reviews for KAI Access app from Google Play Store",
)
async def scrape_reviews(
    request: ScrapeRequest,
    scraper: Scraper,
) -> JSONResponse:
    """Scrape reviews from Google Play Store.

    Args:
        request: Scraping parameters.
        scraper: Scraper service instance.

    Returns:
        JSONResponse with scraping result.
    """
    # Validate sort_by
    valid_sorts = ["newest", "rating", "relevance", "helpful"]
    if request.sort_by not in valid_sorts:
        return JSONResponse(
            content=ApiResponse.error(
                message=f"Invalid sort_by. Must be one of: {valid_sorts}",
                code="INVALID_SORT",
            ),
            status_code=400,
        )

    # Scrape and save
    result = scraper.scrape_and_save(
        max_reviews=request.max_reviews,
        sort_by=request.sort_by,
    )

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="SCRAPER_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message=f"Successfully scraped {result.value['scraped']} reviews",
        ),
    )


@router.get(
    "/status",
    summary="Get scraper status",
    description="Get information about scraped reviews data",
)
async def get_scraper_status(
    scraper: Scraper,
) -> JSONResponse:
    """Get scraper status.

    Args:
        scraper: Scraper service instance.

    Returns:
        JSONResponse with status info.
    """
    result = scraper.get_scrape_status()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(message=result.error, code="STATUS_ERROR"),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data=result.value,
            message="Status retrieved successfully",
        ),
    )
