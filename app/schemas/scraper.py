"""Scraper Pydantic schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ScrapeRequest(BaseModel):
    """Schema for scraping requests."""

    max_reviews: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum number of reviews to scrape (10-1000)",
    )
    sort_by: str = Field(
        default="newest",
        description="Sort order: newest, rating, relevance, helpful",
    )


class ScrapeResponse(BaseModel):
    """Schema for scraping response."""

    scraped: int = Field(..., description="Number of reviews scraped")
    saved_as: str = Field(..., description="Path to saved CSV file")
    app_id: str = Field(..., description="Google Play Store app ID")
    sort_by: str = Field(..., description="Sort order used")


class ScraperStatusResponse(BaseModel):
    """Schema for scraper status response."""

    has_data: bool = Field(..., description="Whether scraped data exists")
    total_reviews: int = Field(..., description="Total number of reviews")
    sentiment_distribution: dict[str, int] = Field(
        default={},
        description="Distribution of sentiments",
    )
    file_path: str | None = Field(None, description="Path to data file")
    last_modified: str | None = Field(None, description="Last modification time")


class DependencyCheckResponse(BaseModel):
    """Schema for dependency check response."""

    has_node: bool = Field(..., description="Whether Node.js is installed")
    has_scraper: bool = Field(..., description="Whether google-play-scraper is installed")
    node_version: str | None = Field(None, description="Node.js version")
    message: str = Field(..., description="Status message")


class InstallDependencyResponse(BaseModel):
    """Schema for install dependency response."""

    success: bool = Field(..., description="Whether installation succeeded")
    message: str = Field(..., description="Installation message")
