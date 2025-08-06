import asyncio
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache
from typing import AsyncGenerator, Generator, Optional

import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from ..core.exceptions import DatabaseError
from ..core.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self) -> None:
        """Initialize the database manager."""
        self.settings = get_settings()
        self._engine = None
        self._async_engine = None
        self._session_factory = None
        self._async_session_factory = None
        self._is_initialized = False
    
    def initialize(self) -> None:
        """Initialize database connections and session factories."""
        if self._is_initialized:
            return
        
        try:
            # Create synchronous engine
            self._engine = create_engine(
                self.settings.database_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                echo=self.settings.debug and self.settings.is_development,
                pool_pre_ping=True,  # Validate connections before use
            )
            
            # Create asynchronous engine
            async_url = self.settings.database_url.replace("postgresql://", "postgresql+asyncpg://")
            self._async_engine = create_async_engine(
                async_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow,
                echo=self.settings.debug and self.settings.is_development,
                pool_pre_ping=True,
            )
            
            # Create session factories
            self._session_factory = sessionmaker(
                bind=self._engine,
                expire_on_commit=False,
            )
            
            self._async_session_factory = async_sessionmaker(
                bind=self._async_engine,
                expire_on_commit=False,
            )
            
            self._is_initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def close(self) -> None:
        """Close all database connections."""
        if self._engine:
            self._engine.dispose()
            logger.debug("Synchronous database engine disposed")
        
        if self._async_engine:
            # Note: async engine disposal should be done in async context
            # This is for graceful shutdown scenarios
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._async_engine.dispose())
                else:
                    loop.run_until_complete(self._async_engine.dispose())
            except Exception as e:
                logger.warning(f"Could not dispose async engine: {e}")
        
        self._is_initialized = False
        logger.info("Database manager closed")
    
    @property
    def engine(self):
        """Get the synchronous database engine."""
        if not self._is_initialized:
            self.initialize()
        return self._engine
    
    @property
    def async_engine(self):
        """Get the asynchronous database engine."""
        if not self._is_initialized:
            self.initialize()
        return self._async_engine
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a synchronous database session.
        
        Yields:
            Session: SQLAlchemy session
        """
        if not self._is_initialized:
            self.initialize()
        
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get an asynchronous database session.
        
        Yields:
            AsyncSession: Async SQLAlchemy session
        """
        if not self._is_initialized:
            self.initialize()
        
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Async database session error: {e}")
            raise DatabaseError(f"Async database operation failed: {e}")
        finally:
            await session.close()
    
    def check_connection(self) -> bool:
        """
        Check if database connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def check_async_connection(self) -> bool:
        """
        Check if async database connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise
        """
        try:
            async with self.get_async_session() as session:
                await session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Async database health check failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        Get database connection information.
        
        Returns:
            dict: Connection information
        """
        if not self._is_initialized:
            return {"status": "not_initialized"}
        
        try:
            with self.get_session() as session:
                result = session.execute(text("""
                    SELECT 
                        current_database() as database_name,
                        current_user as user_name,
                        version() as version,
                        now() as current_time
                """))
                row = result.fetchone()
                
                return {
                    "status": "connected",
                    "database_name": row[0],
                    "user_name": row[1],
                    "version": row[2],
                    "current_time": row[3],
                }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {"status": "error", "error": str(e)}


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


@lru_cache()
def get_database_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    
    Returns:
        DatabaseManager: The database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Convenience functions for common operations
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session (convenience function)."""
    db_manager = get_database_manager()
    return db_manager.get_session()


async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session (convenience function)."""
    db_manager = get_database_manager()
    async with db_manager.get_async_session() as session:
        yield session


def check_database_health() -> bool:
    """Check database health (convenience function)."""
    db_manager = get_database_manager()
    return db_manager.check_connection()


async def check_async_database_health() -> bool:
    """Check async database health (convenience function)."""
    db_manager = get_database_manager()
    return await db_manager.check_async_connection()


# Database initialization for application startup
def initialize_database() -> None:
    """Initialize the database connection."""
    db_manager = get_database_manager()
    db_manager.initialize()
    
    # Verify connection
    if not db_manager.check_connection():
        raise DatabaseError("Failed to establish database connection")
    
    logger.info("Database initialized and connection verified")


def shutdown_database() -> None:
    """Shutdown the database connection."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
        _db_manager = None
    logger.info("Database connections closed")
