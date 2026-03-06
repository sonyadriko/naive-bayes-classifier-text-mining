"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Naive Bayes Classifier API", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: Literal["development", "production", "testing"] = Field(
        default="development", alias="ENVIRONMENT"
    )

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Database
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=3306, alias="DB_PORT")
    db_user: str = Field(default="root", alias="DB_USER")
    db_password: str = Field(default="", alias="DB_PASSWORD")
    db_name: str = Field(default="naive_bayes", alias="DB_NAME")

    # Security
    secret_key: str = Field(default="change-this-secret-key", alias="SECRET_KEY")
    algorithm: str = Field(default="HS256", alias="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        alias="CORS_ORIGINS",
    )

    # File Storage
    data_dir: str = Field(default="data", alias="DATA_DIR")
    upload_dir: str = Field(default="uploads", alias="UPLOAD_DIR")
    allowed_extensions: list[str] = Field(
        default=[".xlsx", ".xls"], alias="ALLOWED_EXTENSIONS"
    )

    # Model
    model_cache_ttl: int = Field(default=3600, alias="MODEL_CACHE_TTL")
    target_column: str = Field(
        default="Durasi Mendapat Kerja", alias="TARGET_COLUMN"
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: str | list[str]) -> list[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.strip("[]").split(",")]
        return v

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v: str) -> str:
        """Validate that secret key is not the default in production."""
        if v in ("change-this-secret-key", ""):
            raise ValueError(
                "SECRET_KEY must be set to a secure value in production"
            )
        return v

    @property
    def database_url(self) -> str:
        """Generate SQLAlchemy database URL."""
        return f"mysql+pymysql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings: Cached application settings.
    """
    return Settings()
