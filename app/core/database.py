"""Database connection and session management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.debug,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Get database session for dependency injection.

    Yields:
        Database session.

    Example:
        ```python
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Get database session as a context manager.

    Yields:
        Database session.

    Example:
        ```python
        with get_db_context() as db:
            user = db.query(User).first()
        ```
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables.

    This creates all tables defined in the models.
    """
    from app.models.user import User

    User.metadata.create_all(bind=engine)
