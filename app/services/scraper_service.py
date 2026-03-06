"""Google Play Store review scraper service.

This module provides functionality to scrape reviews from the Google Play Store
for the KAI Access application using the Python google-play-scraper-rw library.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Try to import the Python library
try:
    from google_play_scraper import Sort, reviews
    HAS_SCRAPER_LIB = True
except ImportError:
    HAS_SCRAPER_LIB = False

from app.core.config import get_settings
from app.utils.response import Err, Ok, Result

settings = get_settings()


class GooglePlayScraper:
    """Scraper for Google Play Store reviews.

    Uses google-play-scraper-rw (Python library) to fetch reviews.
    """

    # KAI Access app ID on Google Play Store
    APP_ID = "com.kai.kaiticketing"

    def __init__(self) -> None:
        """Initialize scraper service."""
        self.data_dir = Path(settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def scrape_reviews(
        self,
        max_reviews: int = 100,
        sort_by: str = "newest",
    ) -> Result[list[dict[str, Any]], str]:
        """Scrape reviews from Google Play Store.

        Args:
            max_reviews: Maximum number of reviews to scrape.
            sort_by: Sort order - "newest", "rating", "relevance", "helpful".

        Returns:
            Ok(list of review dicts) if successful, Err with message if failed.
        """
        if not HAS_SCRAPER_LIB:
            return Err(
                "google-play-scraper library not found. "
                "Run: pip install google-play-scraper"
            )

        try:
            # Validate sort_by
            valid_sorts = ["newest", "rating", "relevance", "helpful"]
            if sort_by not in valid_sorts:
                return Err(f"Invalid sort_by. Must be one of: {valid_sorts}")

            # Map sort options to Sort enum
            sort_map = {
                "newest": Sort.NEWEST,
                "rating": Sort.RATING,
                "relevance": Sort.MOST_RELEVANT,
                "helpful": Sort.NEWEST,  # Fallback to newest since HELPFUL doesn't exist
            }

            # Call the Python library
            result = reviews(
                self.APP_ID,
                lang="id",  # Indonesian
                country="id",
                sort=sort_map.get(sort_by, Sort.NEWEST),
                count=max_reviews,
            )

            # The library returns a tuple: (reviews, continuation_token)
            # We only need the reviews list
            reviews_data = result[0] if isinstance(result, tuple) else result

            if not isinstance(reviews_data, list):
                return Err(f"Unexpected response format: {type(reviews_data).__name__}")

            return Ok(reviews_data)

        except Exception as e:
            return Err(f"Scraping failed: {str(e)}")

    def save_to_csv(
        self,
        reviews: list[dict[str, Any]],
        filename: str | None = None,
    ) -> Result[str, str]:
        """Save scraped reviews to CSV file.

        Args:
            reviews: List of review dictionaries.
            filename: Optional custom filename.

        Returns:
            Ok(file_path) if successful, Err with message if failed.
        """
        try:
            if not reviews:
                return Err("No reviews to save")

            # Normalize data to DataFrame
            df = pd.DataFrame(reviews)

            # Select and rename relevant columns
            column_mapping = {
                "userName": "user_name",
                "content": "content",
                "score": "score",
                "date": "date",
                "thumbsUp": "thumbs_up",
                "replyContent": "reply_content",
                "repliedAt": "replied_at",
            }

            # Keep only columns that exist
            available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df[list(available_columns.keys())].rename(columns=available_columns)

            # Add sentiment labels
            df["sentiment"] = df["score"].apply(self._score_to_sentiment)

            # Remove neutral reviews (score 3) for binary classification
            df = df[df["sentiment"].notna()].copy()

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"reviews_{timestamp}.csv"

            # Save to data directory
            file_path = self.data_dir / filename
            df.to_csv(file_path, index=False, encoding="utf-8")

            # Also save as reviews.csv for convenience
            default_path = self.data_dir / "reviews.csv"
            df.to_csv(default_path, index=False, encoding="utf-8")

            return Ok(str(file_path))

        except Exception as e:
            return Err(f"Failed to save reviews: {str(e)}")

    def scrape_and_save(
        self,
        max_reviews: int = 100,
        sort_by: str = "newest",
    ) -> Result[dict[str, Any], str]:
        """Scrape reviews and save to CSV in one operation.

        Args:
            max_reviews: Maximum number of reviews to scrape.
            sort_by: Sort order.

        Returns:
            Ok(dict with scrape results) if successful, Err with message if failed.
        """
        # Scrape reviews
        scrape_result = self.scrape_reviews(max_reviews, sort_by)
        if scrape_result.is_err():
            return Err(f"Scraping failed: {scrape_result.error}")

        reviews = scrape_result.value

        # Save to CSV
        save_result = self.save_to_csv(reviews)
        if save_result.is_err():
            return Err(f"Saving failed: {save_result.error}")

        file_path = save_result.value

        return Ok(
            {
                "scraped": len(reviews),
                "saved_as": file_path,
                "app_id": self.APP_ID,
                "sort_by": sort_by,
            }
        )

    def get_scrape_status(self) -> Result[dict[str, Any], str]:
        """Get status of scraped data.

        Returns:
            Ok(dict with status info) if successful, Err with message if failed.
        """
        try:
            data_path = self.data_dir / "reviews.csv"

            if not data_path.exists():
                return Ok(
                    {
                        "has_data": False,
                        "total_reviews": 0,
                        "last_updated": None,
                    }
                )

            df = pd.read_csv(data_path)

            # Count by sentiment
            sentiment_counts = df["sentiment"].value_counts().to_dict() if "sentiment" in df.columns else {}

            return Ok(
                {
                    "has_data": True,
                    "total_reviews": len(df),
                    "sentiment_distribution": sentiment_counts,
                    "file_path": str(data_path),
                    "last_modified": datetime.fromtimestamp(data_path.stat().st_mtime).isoformat(),
                }
            )

        except Exception as e:
            return Err(f"Failed to get status: {str(e)}")

    def check_library_status(self) -> dict[str, Any]:
        """Check if the Python scraper library is installed.

        Returns:
            Dict with library status information.
        """
        return {
            "has_library": HAS_SCRAPER_LIB,
            "library_name": "google-play-scraper",
        }

    @staticmethod
    def _score_to_sentiment(score: float | int | None) -> str | None:
        """Convert review score to sentiment label.

        Args:
            score: Review score (1-5).

        Returns:
            "Positif" for score 4-5, "Negatif" for score 1-2, None for score 3 or missing.
        """
        try:
            if pd.isna(score):
                return None
            score_val = int(float(score))
            if score_val >= 4:
                return "Positif"
            elif score_val <= 2:
                return "Negatif"
            else:
                return None  # Neutral (score 3) - excluded for binary classification
        except (ValueError, TypeError):
            return None
