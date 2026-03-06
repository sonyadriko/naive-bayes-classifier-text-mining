"""Base repository with generic CRUD operations.

This module provides a base repository class with common CRUD operations
to avoid repeating code (DRY principle). All repositories should extend
this base class.
"""

from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Session

from app.utils.response import Err, Ok, Result

T = TypeVar("T")


class BaseRepository(Generic[T]):
    """Base repository with generic CRUD operations.

    Provides reusable methods for common database operations.
    Specific repositories can extend this and add custom queries.

    Example:
        ```python
        class UserRepository(BaseRepository[User]):
            def __init__(self, session: Session):
                super().__init__(User, session)

            def find_by_email(self, email: str) -> Result[User | None, str]:
                # Custom query specific to User
                ...
        ```
    """

    def __init__(self, model: type[T], session: Session) -> None:
        """Initialize base repository.

        Args:
            model: SQLAlchemy model class.
            session: Database session.
        """
        self.model = model
        self.session = session

    def get(self, id: int) -> Result[T, str]:
        """Get entity by ID.

        Args:
            id: Entity ID.

        Returns:
            Result containing entity or error message.
        """
        entity = self.session.query(self.model).filter(self.model.id == id).first()
        if not entity:
            return Err(f"{self.model.__name__} with id {id} not found")
        return Ok(entity)

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> Result[list[T], str]:
        """Get all entities with pagination.

        Args:
            skip: Number of records to skip.
            limit: Maximum number of records to return.

        Returns:
            Result containing list of entities or error message.
        """
        entities = self.session.query(self.model).offset(skip).limit(limit).all()
        return Ok(entities)

    def count(self) -> Result[int, str]:
        """Count all entities.

        Returns:
            Result containing count or error message.
        """
        count = self.session.query(self.model).count()
        return Ok(count)

    def create(self, **kwargs: Any) -> Result[T, str]:
        """Create new entity.

        Args:
            **kwargs: Entity field values.

        Returns:
            Result containing created entity or error message.
        """
        try:
            entity = self.model(**kwargs)
            self.session.add(entity)
            self.session.commit()
            self.session.refresh(entity)
            return Ok(entity)
        except Exception as e:
            self.session.rollback()
            return Err(f"Failed to create {self.model.__name__}: {str(e)}")

    def update(self, id: int, **kwargs: Any) -> Result[T, str]:
        """Update entity by ID.

        Args:
            id: Entity ID.
            **kwargs: Fields to update.

        Returns:
            Result containing updated entity or error message.
        """
        try:
            entity_result = self.get(id)
            if entity_result.is_err():
                return entity_result

            entity = entity_result.value
            for key, value in kwargs.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)

            self.session.commit()
            self.session.refresh(entity)
            return Ok(entity)
        except Exception as e:
            self.session.rollback()
            return Err(f"Failed to update {self.model.__name__}: {str(e)}")

    def delete(self, id: int) -> Result[bool, str]:
        """Delete entity by ID.

        Args:
            id: Entity ID.

        Returns:
            Result containing True or error message.
        """
        try:
            entity_result = self.get(id)
            if entity_result.is_err():
                return entity_result

            entity = entity_result.value
            self.session.delete(entity)
            self.session.commit()
            return Ok(True)
        except Exception as e:
            self.session.rollback()
            return Err(f"Failed to delete {self.model.__name__}: {str(e)}")

    def exists(self, id: int) -> Result[bool, str]:
        """Check if entity exists by ID.

        Args:
            id: Entity ID.

        Returns:
            Result containing True/False or error message.
        """
        exists = (
            self.session.query(self.model)
            .filter(self.model.id == id)
            .first()
            is not None
        )
        return Ok(exists)

    def find_by(
        self,
        **filters: Any,
    ) -> Result[list[T], str]:
        """Find entities matching filters.

        Args:
            **filters: Field filters.

        Returns:
            Result containing list of matching entities or error message.
        """
        try:
            query = self.session.query(self.model)
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            entities = query.all()
            return Ok(entities)
        except Exception as e:
            return Err(f"Failed to find {self.model.__name__}: {str(e)}")

    def find_one_by(
        self,
        **filters: Any,
    ) -> Result[T | None, str]:
        """Find first entity matching filters.

        Args:
            **filters: Field filters.

        Returns:
            Result containing entity or None if not found, or error message.
        """
        try:
            query = self.session.query(self.model)
            for key, value in filters.items():
                if hasattr(self.model, key):
                    query = query.filter(getattr(self.model, key) == value)
            entity = query.first()
            return Ok(entity)
        except Exception as e:
            return Err(f"Failed to find {self.model.__name__}: {str(e)}")
