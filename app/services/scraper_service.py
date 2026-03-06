"""Google Play Store review scraper service.

This module provides functionality to scrape reviews from the Google Play Store
for the KAI Access application.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import get_settings
from app.utils.response import Err, Ok, Result

settings = get_settings()


class GooglePlayScraper:
    """Scraper for Google Play Store reviews.

    Uses google-play-scraper (npm package) via subprocess to fetch reviews.
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
            sort_by: Sort order - "newest", "rating", "relevance".

        Returns:
            Ok(list of review dicts) if successful, Err with message if failed.
        """
        try:
            # Validate sort_by
            valid_sorts = ["newest", "rating", "relevance", "helpful"]
            if sort_by not in valid_sorts:
                return Err(f"Invalid sort_by. Must be one of: {valid_sorts}")

            # Map sort options to google-play-scraper constants
            sort_map = {
                "newest": "gplay.sort.NEWEST",
                "rating": "gplay.sort.RATING",
                "relevance": "gplay.sort.RELEVANCE",
                "helpful": "gplay.sort.HELPFULNESS",
            }

            # Create Node.js script
            script = f'''
const gplay = require('google-play-scraper');

gplay.reviews({{
    appId: '{self.APP_ID}',
    sort: {sort_map.get(sort_by, sort_map["newest"])},
    num: {max_reviews},
    paginate: true
}})
.then(response => {{
    // Extract reviews from response
    const reviews = response.data || response;
    console.log(JSON.stringify(reviews));
}})
.catch(error => {{
    console.error(JSON.stringify({{error: error.message}}));
    process.exit(1);
}});
'''

            # Execute Node.js script
            result = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=Path(__file__).parent.parent.parent,
            )

            if result.returncode != 0:
                error_output = result.stderr.strip()
                if error_output:
                    try:
                        error_data = json.loads(error_output)
                        return Err(f"Scraping failed: {error_data.get('error', error_output)}")
                    except json.JSONDecodeError:
                        pass
                return Err(f"Scraping failed: {result.stderr or 'Unknown error'}")

            # Parse JSON output
            output = result.stdout.strip()
            if not output:
                return Err("No data received from scraper")

            reviews = json.loads(output)

            if not isinstance(reviews, list):
                return Err(f"Unexpected response format: {type(reviews).__name__}")

            return Ok(reviews)

        except subprocess.TimeoutExpired:
            return Err("Scraping timed out. Please try again with fewer reviews.")
        except FileNotFoundError:
            return Err("Node.js not found. Please install Node.js and google-play-scraper.")
        except json.JSONDecodeError as e:
            return Err(f"Failed to parse scraper output: {str(e)}")
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


def check_node_dependencies() -> Result[bool, str]:
    """Check if Node.js and google-play-scraper are installed.

    Returns:
        Ok(True) if all dependencies are available, Err with message if not.
    """
    try:
        # Check Node.js
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            return Err("Node.js is not installed. Please install Node.js from https://nodejs.org/")

        # Check google-play-scraper
        result = subprocess.run(
            ["npm", "list", "google-play-scraper"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode != 0:
            return Err(
                "google-play-scraper is not installed. "
                "Run: npm install google-play-scraper"
            )

        return Ok(True)

    except subprocess.TimeoutExpired:
        return Err("Dependency check timed out")
    except FileNotFoundError:
        return Err("Node.js is not installed. Please install Node.js from https://nodejs.org/")
    except Exception as e:
        return Err(f"Dependency check failed: {str(e)}")


def install_dependencies() -> Result[str, str]:
    """Install google-play-scraper npm package.

    Returns:
        Ok(success message) if successful, Err with message if failed.
    """
    try:
        result = subprocess.run(
            ["npm", "install", "--save", "google-play-scraper"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode != 0:
            return Err(f"Installation failed: {result.stderr}")

        return Ok("google-play-scraper installed successfully")

    except subprocess.TimeoutExpired:
        return Err("Installation timed out")
    except Exception as e:
        return Err(f"Installation failed: {str(e)}")
