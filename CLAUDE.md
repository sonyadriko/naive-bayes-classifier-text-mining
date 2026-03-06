# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Sentiment Analysis API for KAI Access app reviews using a custom Multinomial Naive Bayes Classifier. The codebase follows clean architecture principles with separation of concerns across API, services, repositories, and models layers.

The application now supports:
1. **Text Sentiment Analysis** - Classify Indonesian text reviews as Positif/Negatif
2. **Google Play Store Scraper** - Scrape KAI Access reviews for training data
3. **Legacy Job Placement Prediction** - Original categorical feature prediction (deprecated)

## Common Commands

### Development
```bash
# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Install Python dependencies
pip install -r dependencies.txt
pip install -r dependencies-dev.txt  # dev dependencies

# Install npm dependencies (for scraper)
npm install
```

### Testing
```bash
pytest                              # Run all tests
pytest tests/test_auth.py           # Run specific test file
pytest --cov=app --cov-report=html  # With coverage
```

### Docker (via Make)
```bash
make up       # Start development environment
make down     # Stop development environment
make test     # Run tests in container
make shell    # Open shell in API container
```

## Architecture

### Layer Structure
- **API Layer** (`app/api/v1/`): FastAPI routes with dependency injection
- **Service Layer** (`app/services/`): Business logic, returns Result types
- **Repository Layer** (`app/repositories/`): Database access via SQLAlchemy
- **Models** (`app/models/`): SQLAlchemy ORM models + Naive Bayes classifiers
- **Schemas** (`app/schemas/`): Pydantic models for request/response validation
- **Utils** (`app/utils/`): Text preprocessing, response builders

### Key Patterns

**Result Type for Error Handling**
Services return `Result[T, str]` instead of raising exceptions:
```python
return Ok(data)  # Success
return Err("error message")  # Failure
# Usage: result.is_ok(), result.is_err(), result.value, result.error
```

**Standardized API Responses**
All endpoints use `ApiResponse` builder:
```python
return JSONResponse(ApiResponse.success(data=result, message="Success"))
return JSONResponse(ApiResponse.error(message="Not found", code="NOT_FOUND", status_code=404))
```

**Dependency Injection**
Type aliases defined in `dependencies.py` and `app/api/v1/router.py`:
- `DBSession`: Injected database session
- `CurrentUser`: Authenticated user from JWT token

### Sentiment Analysis Pipeline

**Text Preprocessing** (`app/utils/text_preprocessing.py`)
```python
from app.utils.text_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
result = preprocessor.preprocess("Aplikasinya SANGAT bagus!!")
# Returns: "aplikasi bagus"
```

Pipeline:
1. Case Folding - Convert to lowercase
2. Cleaning - Remove URLs, mentions, special chars, numbers
3. Tokenization - Split into words
4. Stopword Removal - Remove Indonesian stopwords
5. Normalization - Convert informal words (yg -> yang)
6. Stemming - Reduce to root form using Sastrawi

**Multinomial Naive Bayes** (`app/models/naive_bayes.py`)
```python
from app.models.naive_bayes import MultinomialNaiveBayes

classifier = MultinomialNaiveBayes(alpha=1.0)
classifier.train(texts, labels, vocabulary)
prediction = classifier.predict("aplikasi bagus")
```

### Google Play Store Scraper

**Scraper Service** (`app/services/scraper_service.py`)
```python
from app.services.scraper_service import GooglePlayScraper

scraper = GooglePlayScraper()
result = scraper.scrape_and_save(max_reviews=100, sort_by="newest")
```

Uses `google-play-scraper` (npm) via subprocess. KAI Access app ID: `com.kai.kaiticketing`

### Configuration

Settings via `app/core/config.py` using Pydantic Settings:
- Loads from `.env` file
- Access via `get_settings()` singleton
- Environment: `development`, `production`, or `testing`

### Authentication

- JWT-based with bcrypt password hashing
- Token in `Authorization: Bearer <token>` header
- `get_current_user()` dependency extracts user from token

## API Endpoints

### Scraper
```
POST /api/v1/scraper/reviews
  Body: { "max_reviews": 100, "sort_by": "newest" }
  Response: { "data": { "scraped": 100, "saved_as": "reviews_xxx.csv" } }

GET /api/v1/scraper/status
  Response: { "data": { "total_reviews": 1500, "sentiment_distribution": {...} } }

GET /api/v1/scraper/dependencies
  Response: { "data": { "has_node": true, "has_scraper": true } }

POST /api/v1/scraper/install
  Response: { "data": { "installed": true } }
```

### Sentiment Prediction
```
POST /api/v1/predictions/sentiment
  Body: { "text": "aplikasi sangat bagus" }
  Response: { "data": { "predicted_class": "Positif", "confidence": 0.85 } }

POST /api/v1/predictions/sentiment/batch
  Body: { "texts": ["text1", "text2"] }
  Response: { "data": { "predictions": [...] } }

POST /api/v1/predictions/sentiment/train
  Body: { "force_retrain": false }
  Response: { "data": { "model_type": "MultinomialNaiveBayes", ... } }

GET /api/v1/predictions/preprocess/sample
  Query: ?text=Sample text
  Response: Preprocessing steps visualization
```

### Data Upload
```
POST /api/v1/data/upload
  Upload CSV/Excel with review data (columns: content, score)
```

### Web Routes
```
GET /scraper     - Scraper UI page
GET /prediction  - Sentiment analysis UI page
GET /data        - Data upload UI page
GET /evaluation  - Model evaluation UI page
```

## Data Format

### Training CSV (Google Play Store Reviews)
```csv
content,score,userName
"aplikasi sangat membantu",5,"User1"
"selalu error bos",1,"User2"
```

### Labeling Rule (score -> sentiment)
- Score 4-5 → **Positif**
- Score 1-2 → **Negatif**
- Score 3 → **Netral** (excluded for binary classification)

### Preprocessed CSV Output
```csv
content,preprocessed,sentiment
"aplikasi sangat membantu","aplikasi bantu","Positif"
"selalu error bos","error bos","Negatif"
```

## File Structure

```
naive-bayes-classifier/
├── main.py                  # FastAPI app entry point
├── dependencies.py          # Dependency injection providers
├── package.json             # Node.js dependencies (google-play-scraper)
├── Makefile                 # Docker commands
│
├── app/
│   ├── api/v1/
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── data.py          # Data upload endpoints
│   │   ├── evaluation.py    # Model evaluation endpoints
│   │   ├── predictions.py   # Sentiment prediction endpoints
│   │   ├── router.py        # Router aggregation
│   │   ├── scraper.py       # Google Play Store scraper endpoints
│   │   └── users.py         # User CRUD endpoints
│   ├── core/
│   │   ├── config.py        # Pydantic Settings (loads .env)
│   │   ├── database.py      # Database connection setup
│   │   └── security.py      # JWT/password utilities
│   ├── middleware/
│   │   └── error_handler.py # Global error handling
│   ├── models/
│   │   ├── user.py          # User ORM model
│   │   └── naive_bayes.py   # NaiveBayesClassifier (categorical) + MultinomialNaiveBayes (text)
│   ├── repositories/
│   │   ├── base.py          # Generic repository base class
│   │   └── user_repository.py
│   ├── schemas/
│   │   ├── auth.py
│   │   ├── data.py
│   │   ├── prediction.py    # Sentiment + categorical schemas
│   │   ├── scraper.py       # Scraper request/response schemas
│   │   └── user.py
│   ├── services/
│   │   ├── auth_service.py
│   │   ├── data_service.py  # Data processing + text dataset methods
│   │   ├── label_service.py # Feature labeling utilities
│   │   ├── model_service.py # Legacy categorical model service
│   │   ├── scraper_service.py # Google Play Store scraper
│   │   ├── sentiment_service.py # Sentiment analysis service
│   │   └── user_service.py
│   └── utils/
│       ├── response.py      # Result type (Ok/Err)
│       └── text_preprocessing.py # Indonesian text preprocessing pipeline
│
├── data/
│   ├── dictionaries/
│   │   └── normalisasi.txt  # Indonesian word normalization dict
│   ├── stopwords/
│   │   └── stopword.txt     # Indonesian stopwords list
│   ├── models/              # Cached trained models
│   ├── reviews/             # Scraped review CSV files
│   └── encoders/            # Label encoders for categorical features
│
├── templates/
│   ├── base.html
│   ├── prediction.html      # Sentiment analysis UI
│   ├── scraper.html         # Scraper UI
│   ├── data.html            # Data upload UI
│   └── evaluation.html      # Model evaluation UI
│
└── tests/
    ├── conftest.py
    ├── test_auth.py
    ├── test_data.py
    ├── test_predictions.py
    └── test_users.py
```

## External Dependencies

### Python
- `fastapi` - Web framework
- `pandas` - Data processing
- `scikit-learn` - Evaluation metrics
- `sastrawi` - Indonesian stemming
- `nltk` - NLP utilities (stopwords)

### Node.js (for scraper)
- `google-play-scraper` - Google Play Store reviews scraper

## Legacy Features (Deprecated)

The original job placement prediction using categorical features is preserved for reference:
- `NaiveBayesClassifier` - Gaussian/categorical Naive Bayes
- `ModelService` - Model service for categorical features
- Categorical prediction endpoints remain in `/predictions/predict`

## Target Column

For sentiment analysis, the target is sentiment derived from review scores:
- Score 4-5 → Positif
- Score 1-2 → Negatif

## Data Directory Structure

```
data/
├── dictionaries/
│   └── normalisasi.txt    # Indonesian word normalization dict
├── stopwords/
│   └── stopword.txt       # Indonesian stopwords list
├── models/                # Cached trained models (.pkl files)
├── reviews/               # Scraped review data (.csv files)
└── encoders/              # Label encoders for categorical features
```

## Data Upload

CSV/Excel files are uploaded to `uploads/` directory and processed via `app/services/data_service.py`. Training data is stored in `data/` directory.

For sentiment analysis:
- Upload CSV with `content` column (review text)
- `score` column is optional (used for auto-labeling)
- Or use the scraper to fetch reviews directly

## Normalization Dictionary

The `normalisasi.txt` file contains informal-to-formal Indonesian word mappings (format: `informal,formal` per line). It is loaded by `TextPreprocessor` from `data/dictionaries/normalisasi.txt`. Examples:
- `yg,yang` - "yg" → "yang"
- `gak,tidak` - "gak" → "tidak"
- `apk,aplikasi` - "apk" → "aplikasi"
