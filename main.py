"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.database import init_db
from app.middleware.error_handler import register_error_handlers
from app.utils.response import ApiResponse

settings = get_settings()

# Create required directories
os.makedirs(settings.data_dir, exist_ok=True)
os.makedirs(settings.upload_dir, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler.

    Initializes database on startup and performs cleanup on shutdown.
    """
    # Startup
    if settings.debug:
        init_db()
    yield
    # Shutdown - cleanup if needed


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for predicting vocational school students' job placement outcomes using Naive Bayes Classifier",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Register error handlers
register_error_handlers(app)

# Include API router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root() -> JSONResponse:
    """Root endpoint with API information.

    Returns:
        API information response.
    """
    return JSONResponse(
        ApiResponse.success(
            data={
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment,
                "docs": "/docs",
            },
            message="Welcome to Naive Bayes Classifier API",
        )
    )


@app.get("/health")
async def health_check() -> JSONResponse:
    """Health check endpoint.

    Returns:
        Health status response.
    """
    return JSONResponse(
        ApiResponse.success(data={"status": "healthy"}, message="Service is running")
    )


# ============================================================================
# Web Routes (HTML Pages)
# ============================================================================


@app.get("/login")
async def login_page(request: Request):
    """Render login/register page.

    Args:
        request: FastAPI request object.

    Returns:
        Template response with login page.
    """
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Render dashboard page.

    Args:
        request: FastAPI request object.

    Returns:
        Template response with dashboard page.
    """
    # Get user from token if available (optional - for SSR)
    user = None
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        # For SSR, you could decode the JWT here to get user info
        # For now, the frontend will handle auth via localStorage
        pass

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "user": user}
    )


@app.get("/prediction")
async def prediction_page(request: Request):
    """Render prediction page.

    Args:
        request: FastAPI request object.

    Returns:
        Template response with prediction page.
    """
    return templates.TemplateResponse(
        "prediction.html",
        {"request": request, "user": None}
    )


@app.get("/data")
async def data_page(request: Request):
    """Render data upload page.

    Args:
        request: FastAPI request object.

    Returns:
        Template response with data page.
    """
    return templates.TemplateResponse(
        "data.html",
        {"request": request, "user": None}
    )


@app.get("/evaluation")
async def evaluation_page(request: Request):
    """Render evaluation page.

    Args:
        request: FastAPI request object.

    Returns:
        Template response with evaluation page.
    """
    return templates.TemplateResponse(
        "evaluation.html",
        {"request": request, "user": None}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
