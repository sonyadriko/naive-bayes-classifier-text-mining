# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI backend for predicting vocational school students' job placement outcomes using a custom Naive Bayes Classifier. The codebase follows clean architecture principles with separation of concerns across API, services, repositories, and models layers.

## Common Commands

### Development
```bash
# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Install dependencies
pip install -r dependencies.txt
pip install -r dependencies-dev.txt  # dev dependencies
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
- **Models** (`app/models/`): SQLAlchemy ORM models + NaiveBayesClassifier
- **Schemas** (`app/schemas/`): Pydantic models for request/response validation

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
return JSONResponse(ApiResponse.success(data=user, message="User created"))
return JSONResponse(ApiResponse.error(message="Not found", code="NOT_FOUND", status_code=404))
```

**Dependency Injection**
Type aliases defined in `dependencies.py` and `app/api/v1/router.py`:
- `DBSession`: Injected database session
- `CurrentUser`: Authenticated user from JWT token

### Naive Bayes Classifier

Custom implementation in `app/models/naive_bayes.py`:
- `NaiveBayesClassifier.train(X, y)`: Train on DataFrame
- `NaiveBayesClassifier.predict(features)`: Predict from dict
- `ModelCache`: File-based pickle caching (stored in `data/models/`)
- Model is cached after training to avoid retraining on every request

### Configuration

Settings via `app/core/config.py` using Pydantic Settings:
- Loads from `.env` file
- Access via `get_settings()` singleton
- Environment: `development`, `production`, or `testing`

### Authentication

- JWT-based with bcrypt password hashing
- Token in `Authorization: Bearer <token>` header
- `get_current_user()` dependency extracts user from token

## Target Column

The prediction target is `Durasi Mendapat Kerja` (Duration to Get Job) with values like `< 3 bulan`, `3-6 bulan`, `> 6 bulan`. This is configured via `TARGET_COLUMN` env var.

## Data Upload

Excel files (`.xlsx`, `.xls`) are uploaded to `uploads/` directory and processed via `app/services/data_service.py`. Training data is stored in `data/` directory.
