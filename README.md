# KAI Access Sentiment Analysis API

A FastAPI backend for sentiment analysis of KAI Access app reviews using Multinomial Naive Bayes Classifier. The system scrapes reviews from Google Play Store and classifies them as **Positif** (Positive) or **Negatif** (Negative).

## Features

- **Text Sentiment Analysis**: Classify Indonesian text reviews using Multinomial Naive Bayes
- **Google Play Store Scraper**: Fetch KAI Access reviews automatically (Python library)
- **Web Dashboard**: Interactive UI for sentiment analysis, scraping, and model evaluation
- **Indonesian Text Preprocessing**: Complete pipeline for Bahasa Indonesia text processing
  - Case folding, cleaning, tokenization
  - Stopword removal
  - Informal word normalization (yg → yang)
  - Stemming with Sastrawi
- **Clean Code Architecture**: Separation of concerns with services, repositories, and schemas
- **JWT Authentication**: Secure token-based authentication with bcrypt password hashing
- **Standardized API Responses**: Consistent `{data, meta}` response format
- **Model Caching**: Train once, predict many times - no need to retrain on every request

## Project Structure

```
naive-bayes-classifier/
├── main.py                 # Application entry point
├── dependencies.py         # Dependency injection providers
├── dependencies.txt        # Python dependencies
├── docker-compose.yml      # Docker compose configuration
├── Dockerfile              # Multi-stage Docker build
├── Makefile                # Docker commands
├── .env                    # Environment configuration
│
├── app/
│   ├── api/v1/            # API routes
│   │   ├── auth.py        # Authentication endpoints
│   │   ├── users.py       # User CRUD endpoints
│   │   ├── data.py        # Data upload endpoints
│   │   ├── predictions.py # Sentiment prediction endpoints
│   │   ├── evaluation.py  # Model evaluation endpoints
│   │   ├── scraper.py     # Scraper endpoints (NEW)
│   │   └── router.py      # Router aggregation
│   │
│   ├── core/              # Core configuration
│   │   ├── config.py      # Pydantic Settings
│   │   ├── security.py    # JWT, password hashing
│   │   └── database.py    # Database connection
│   │
│   ├── models/            # Data models
│   │   ├── user.py        # User model
│   │   └── naive_bayes.py # Naive Bayes classifiers (Multinomial + Categorical)
│   │
│   ├── schemas/           # Pydantic schemas
│   │   ├── user.py        # User schemas
│   │   ├── auth.py        # Auth schemas
│   │   ├── prediction.py  # Prediction schemas
│   │   ├── scraper.py     # Scraper schemas (NEW)
│   │   └── data.py        # Data schemas
│   │
│   ├── services/          # Business logic
│   │   ├── auth_service.py
│   │   ├── user_service.py
│   │   ├── data_service.py
│   │   ├── model_service.py
│   │   ├── sentiment_service.py # Sentiment analysis service (NEW)
│   │   ├── scraper_service.py   # Scraper service (NEW)
│   │   └── label_service.py
│   │
│   ├── repositories/      # Data access layer
│   │   ├── base.py        # Generic CRUD
│   │   └── user_repository.py
│   │
│   ├── middleware/        # Middleware
│   │   └── error_handler.py
│   │
│   └── utils/             # Utilities
│       ├── response.py    # API response builder
│       └── text_preprocessing.py # Indonesian text preprocessing (NEW)
│
├── templates/             # Web UI templates
│   ├── base.html
│   ├── login.html         # Login/register page
│   ├── dashboard.html     # Dashboard page
│   ├── prediction.html    # Sentiment analysis UI
│   ├── scraper.html       # Scraper UI
│   ├── data.html
│   └── evaluation.html
│
└── tests/                 # Test suite
    ├── conftest.py
    ├── test_auth.py
    ├── test_users.py
    ├── test_data.py
    └── test_predictions.py
```

### Data Directory

```
data/
├── dictionaries/
│   └── normalisasi.txt    # Indonesian word normalization (yg→yang, gak→tidak)
├── stopwords/
│   └── stopword.txt       # Indonesian stopwords for filtering
├── models/                # Cached trained models (.pkl)
├── reviews/               # Scraped Google Play Store reviews (.csv)
└── encoders/              # Label encoders for categorical features
```

## Installation

### 1. Python Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Python dependencies (includes google-play-scraper)
pip install -r dependencies.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Database (Optional - for user management)

```bash
mysql -u root -p
CREATE DATABASE naive_bayes;
```

### Docker (Recommended)

```bash
# Build and start all services
make up

# API will be available at http://localhost:8000
# MySQL at localhost:3306
# Adminer (DB UI) at http://localhost:8080 (enable with --profile tools)
```

## Running the Application

### Development

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Commands

```bash
make up           # Start development environment
make down         # Stop development environment
make logs         # View logs from all services
make logs-api     # View API logs
make logs-db      # View MySQL logs
make shell        # Open shell in API container
make test         # Run tests in container
make db-migrate   # Initialize database
make db-shell     # Open MySQL shell
```

## API Documentation

Once running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Web UI:
  - `http://localhost:8000/login` - Login/Register
  - `http://localhost:8000/dashboard` - Dashboard
  - `http://localhost:8000/prediction` - Sentiment Analysis
  - `http://localhost:8000/scraper` - Google Play Scraper
  - `http://localhost:8000/data` - Data Upload
  - `http://localhost:8000/evaluation` - Model Evaluation

## API Endpoints

### Scraper (Google Play Store)

- `POST /api/v1/scraper/reviews` - Scrape KAI Access reviews
- `GET /api/v1/scraper/status` - Get scraped data status

### Sentiment Prediction

- `POST /api/v1/predictions/sentiment` - Analyze single text sentiment
- `POST /api/v1/predictions/sentiment/batch` - Analyze multiple texts
- `POST /api/v1/predictions/sentiment/train` - Train/retrain sentiment model
- `GET /api/v1/predictions/preprocess/sample` - Test text preprocessing

### Data Management

- `POST /api/v1/data/upload` - Upload training data (CSV/Excel)
- `GET /api/v1/data/read` - Read training data
- `GET /api/v1/data/info` - Get dataset information

### Authentication

- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user

### Evaluation

- `POST /api/v1/evaluation/confusion-matrix` - Evaluate model performance

## Text Preprocessing Pipeline

The Indonesian text preprocessing pipeline includes:

1. **Case Folding**: Convert to lowercase
2. **Cleaning**: Remove URLs, mentions, special characters, numbers
3. **Tokenization**: Split into words
4. **Stopword Removal**: Remove common Indonesian words (yang, ada, dll)
5. **Normalization**: Convert informal words (yg → yang, gak → tidak)
6. **Stemming**: Reduce to root form using Sastrawi (aplikasinya → aplikasi)

## Data Format

### Training CSV

```csv
content,score,userName
"aplikasi sangat membantu",5,"User1"
"selalu error bos",1,"User2"
```

### Labeling Rule (score → sentiment)

- Score 4-5 → **Positif**
- Score 1-2 → **Negatif**
- Score 3 → Excluded (neutral)

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py
```

## License

MIT License
