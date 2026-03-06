"""Scraper API endpoints.

Handles Google Play Store review scraping operations.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.schemas.scraper import (
    DependencyCheckResponse,
    InstallDependencyResponse,
    ScrapeRequest,
    ScrapeResponse,
    ScraperStatusResponse,
)
from app.services.scraper_service import GooglePlayScraper, check_node_dependencies, install_dependencies
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


@router.get(
    "/dependencies",
    summary="Check scraper dependencies",
    description="Check if Node.js and google-play-scraper are installed",
)
async def check_dependencies() -> JSONResponse:
    """Check if required dependencies are installed.

    Returns:
        JSONResponse with dependency status.
    """
    import subprocess

    # Check Node.js
    has_node = False
    node_version = None
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_node = result.returncode == 0
        node_version = result.stdout.strip() if has_node else None
    except Exception:
        has_node = False

    # Check google-play-scraper
    has_scraper = False
    try:
        result = subprocess.run(
            ["npm", "list", "google-play-scraper"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent.parent.parent,
        )
        has_scraper = result.returncode == 0
    except Exception:
        has_scraper = False

    status_data = {
        "has_node": has_node,
        "has_scraper": has_scraper,
        "node_version": node_version,
    }

    if has_node and has_scraper:
        message = "All dependencies are installed"
    elif not has_node:
        message = "Node.js is not installed. Please install from https://nodejs.org/"
    else:
        message = "google-play-scraper is not installed. Run POST /scraper/install"

    return JSONResponse(
        content=ApiResponse.success(
            data=status_data,
            message=message,
        ),
    )


@router.post(
    "/install",
    summary="Install scraper dependencies",
    description="Install google-play-scraper npm package",
)
async def install_scraper_dependencies() -> JSONResponse:
    """Install google-play-scraper npm package.

    Returns:
        JSONResponse with installation result.
    """
    result = install_dependencies()

    if result.is_err():
        return JSONResponse(
            content=ApiResponse.error(
                message=result.error,
                code="INSTALL_ERROR",
            ),
            status_code=400,
        )

    return JSONResponse(
        content=ApiResponse.success(
            data={"installed": True},
            message=result.value,
        ),
    )


# Import Path for type checking
from pathlib import Path