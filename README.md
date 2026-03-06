# Naive Bayes Classifier API - FastAPI Backend

A clean, production-ready FastAPI backend for predicting vocational school students' job placement outcomes using Naive Bayes Classifier.

## Features

- **FastAPI Framework**: Modern async Python web framework with automatic OpenAPI documentation
- **Clean Code Architecture**: Separation of concerns with services, repositories, and schemas
- **SQLAlchemy ORM**: Type-safe database operations with MySQL
- **JWT Authentication**: Secure token-based authentication with bcrypt password hashing
- **Standardized API Responses**: Consistent `{data, meta}` response format
- **Error Handling**: Global exception handling with structured error responses
- **Model Caching**: Train once, predict many times - no need to retrain on every request
- **Comprehensive Tests**: Pytest-based test suite with fixtures

## Project Structure

```
backend-new/
├── main.py                 # Application entry point
├── dependencies.py         # Dependency injection providers
├── dependencies.txt        # Production dependencies
├── dependencies-dev.txt    # Development dependencies
├── .env                    # Environment configuration (create from .env.example)
│
├── app/
│   ├── api/v1/            # API routes
│   │   ├── auth.py        # Authentication endpoints
│   │   ├── users.py       # User CRUD endpoints
│   │   ├── data.py        # Data upload/convert/read endpoints
│   │   ├── predictions.py # Prediction endpoints
│   │   ├── evaluation.py  # Model evaluation endpoints
│   │   └── router.py      # Router aggregation
│   │
│   ├── core/              # Core configuration
│   │   ├── config.py      # Pydantic Settings
│   │   ├── security.py    # JWT, password hashing
│   │   └── database.py    # Database connection
│   │
│   ├── models/            # SQLAlchemy models
│   │   ├── user.py        # User model
│   │   └── naive_bayes.py # Naive Bayes classifier
│   │
│   ├── schemas/           # Pydantic schemas
│   │   ├── user.py        # User schemas
│   │   ├── auth.py        # Auth schemas
│   │   ├── prediction.py  # Prediction schemas
│   │   └── data.py        # Data schemas
│   │
│   ├── services/          # Business logic
│   │   ├── auth_service.py
│   │   ├── user_service.py
│   │   ├── data_service.py
│   │   ├── model_service.py
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
│       └── response.py    # API response builder
│
└── tests/                 # Test suite
    ├── conftest.py        # Pytest fixtures
    ├── test_auth.py
    ├── test_users.py
    ├── test_data.py
    └── test_predictions.py
```

## Installation

1. **Create Python virtual environment**:
   ```bash
   cd backend-new
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r dependencies.txt
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Create database**:
   ```bash
   mysql -u root -p
   CREATE DATABASE naive_bayes;
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

## API Documentation

Once running, access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user

### Users (Protected)
- `GET /api/v1/users` - List users (paginated)
- `GET /api/v1/users/me` - Get current user
- `GET /api/v1/users/{id}` - Get user by ID
- `PUT /api/v1/users/{id}` - Update user
- `DELETE /api/v1/users/{id}` - Delete user

### Data
- `POST /api/v1/data/upload` - Upload training data (Excel)
- `POST /api/v1/data/convert` - Convert categorical data
- `GET /api/v1/data/read` - Read training data
- `GET /api/v1/data/labels` - Get label mappings
- `GET /api/v1/data/info` - Get dataset information

### Predictions
- `POST /api/v1/predictions/predict` - Make prediction
- `GET /api/v1/predictions/model/info` - Get model info
- `POST /api/v1/predictions/model/cache/clear` - Clear model cache

### Evaluation
- `POST /api/v1/evaluation/confusion-matrix` - Evaluate model

## Standardized Response Format

All API responses follow this structure:

**Success Response:**
```json
{
  "data": { ... },
  "meta": {
    "status": "success",
    "message": "Operation completed",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

**Paginated Response:**
```json
{
  "data": [ ... ],
  "meta": {
    "status": "success",
    "pagination": {
      "total": 100,
      "page": 1,
      "page_size": 10,
      "total_pages": 10
    },
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

**Error Response:**
```json
{
  "data": null,
  "meta": {
    "status": "error",
    "message": "Error description",
    "code": "ERROR_CODE",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_auth.py
```

## Migration from Flask

### API Endpoint Mapping

| Flask | FastAPI |
|-------|---------|
| `POST /auth/login` | `POST /api/v1/auth/login` |
| `POST /auth/register` | `POST /api/v1/auth/register` |
| `GET /users/` | `GET /api/v1/users` |
| `GET /users/<id>` | `GET /api/v1/users/{id}` |
| `PUT /users/<id>` | `PUT /api/v1/users/{id}` |
| `DELETE /users/<id>` | `DELETE /api/v1/users/{id}` |
| `POST /upload/` | `POST /api/v1/data/upload` |
| `POST /convert/` | `POST /api/v1/data/convert` |
| `GET /read/` | `GET /api/v1/data/read` |
| `GET /get_labels` | `GET /api/v1/data/labels` |
| `POST /predict/` | `POST /api/v1/predictions/predict` |
| `POST /evaluate/confusion_matrix` | `POST /api/v1/evaluation/confusion-matrix` |

### Key Changes

1. **Response Format**: All responses now include `{data, meta}` structure
2. **Authentication**: JWT tokens in `Authorization: Bearer <token>` header
3. **Error Handling**: Structured error responses with error codes
4. **Model Caching**: Model is cached after training - no need to retrain on every prediction
5. **Type Safety**: Full type hints with Pydantic validation

## License

MIT License
# naive-bayes-classifier-text-mining
