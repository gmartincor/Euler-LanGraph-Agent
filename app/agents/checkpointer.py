"""PostgreSQL Checkpointer for LangGraph Workflows.

This module provides professional PostgreSQL-based checkpointing for LangGraph workflows,
reusing the existing database infrastructure to avoid code duplication.

Key Design Patterns Applied:
- Adapter Pattern: Adapts existing DatabaseManager to LangGraph checkpointing interface
- Dependency Injection: Database dependencies injected cleanly
- Single Responsibility: Focus only on checkpoint persistence
- Professional Error Handling: Comprehensive exception management
- DRY Principle: Reuses existing database connection and error handling

Architecture Benefits:
- Zero Code Duplication: Reuses existing DatabaseManager infrastructure
- Professional Quality: Maintains existing database patterns and error handling
- Performance Optimized: Leverages existing connection pooling and transactions
- Secure: Uses existing database security and validation patterns
"""

from typing import Any, Dict, List, Optional, Tuple, AsyncIterator
from datetime import datetime
import json
import asyncio

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from langchain_core.runnables import RunnableConfig

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import DatabaseError, ValidationError
from ..database.connection import DatabaseManager, get_database_manager

logger = get_logger(__name__)


class PostgreSQLCheckpointer(BaseCheckpointSaver):
    """
    Professional PostgreSQL checkpointer for LangGraph workflows.
    
    This checkpointer integrates with the existing database infrastructure
    to provide persistent state management for mathematical reasoning workflows.
    It follows DRY principles by reusing the existing DatabaseManager.
    
    Key Features:
    - Persistent workflow state storage
    - Thread-safe checkpoint operations
    - Integration with existing database infrastructure
    - Professional error handling and logging
    - Optimized for mathematical reasoning workflows
    
    Database Schema:
    - langgraph_checkpoints: Main checkpoint storage
    - langgraph_writes: Write operations log
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize PostgreSQL checkpointer with existing database infrastructure.
        
        Args:
            db_manager: Optional DatabaseManager instance (uses default if None)
        """
        super().__init__()
        self.db_manager = db_manager or get_database_manager()
        self._tables_initialized = False
        
        logger.info("PostgreSQLCheckpointer initialized with existing database infrastructure")
    
    @log_function_call(logger)
    async def _ensure_tables_exist(self) -> None:
        """
        Ensure required checkpoint tables exist in the database.
        
        This method creates the necessary tables for checkpointing if they don't exist,
        reusing the existing database connection and error handling patterns.
        
        Raises:
            DatabaseError: If table creation fails
        """
        if self._tables_initialized:
            return
            
        try:
            await self.db_manager.ensure_connected()
            
            # Create checkpoints table
            checkpoints_sql = """
            CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                thread_id TEXT NOT NULL,
                thread_ts TIMESTAMP NOT NULL,
                parent_ts TIMESTAMP,
                checkpoint JSONB NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, thread_ts)
            );
            
            CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_thread_id 
            ON langgraph_checkpoints(thread_id);
            
            CREATE INDEX IF NOT EXISTS idx_langgraph_checkpoints_created_at 
            ON langgraph_checkpoints(created_at);
            """
            
            # Create writes table
            writes_sql = """
            CREATE TABLE IF NOT EXISTS langgraph_writes (
                thread_id TEXT NOT NULL,
                thread_ts TIMESTAMP NOT NULL,
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                channel TEXT NOT NULL,
                value JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (thread_id, thread_ts, task_id, idx)
            );
            
            CREATE INDEX IF NOT EXISTS idx_langgraph_writes_thread_id 
            ON langgraph_writes(thread_id);
            """
            
            async with self.db_manager.get_connection() as conn:
                await conn.execute(checkpoints_sql)
                await conn.execute(writes_sql)
                await conn.commit()
            
            self._tables_initialized = True
            logger.info("LangGraph checkpoint tables ensured in database")
            
        except Exception as e:
            logger.error(f"Failed to ensure checkpoint tables exist: {e}")
            raise DatabaseError(f"Checkpoint table creation failed: {e}") from e
    
    @log_function_call(logger)
    async def aget_tuple(self, config: RunnableConfig) -> Optional[Tuple[str, Checkpoint]]:
        """
        Get the latest checkpoint for a given thread.
        
        Args:
            config: Runnable configuration containing thread information
            
        Returns:
            Optional tuple of (checkpoint_id, checkpoint) or None if not found
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If config is invalid
        """
        try:
            await self._ensure_tables_exist()
            
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                raise ValidationError("thread_id is required in config.configurable")
            
            query = """
            SELECT thread_ts, checkpoint, metadata
            FROM langgraph_checkpoints 
            WHERE thread_id = $1 
            ORDER BY thread_ts DESC 
            LIMIT 1
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetchrow(query, thread_id)
                
                if not result:
                    logger.debug(f"No checkpoint found for thread_id: {thread_id}")
                    return None
                
                checkpoint_data = {
                    "v": 1,
                    "id": result["thread_ts"].isoformat(),
                    "ts": result["thread_ts"].isoformat(),
                    **json.loads(result["checkpoint"])
                }
                
                checkpoint = Checkpoint(**checkpoint_data)
                
                logger.debug(f"Retrieved checkpoint for thread_id: {thread_id}")
                return (result["thread_ts"].isoformat(), checkpoint)
                
        except Exception as e:
            logger.error(f"Failed to get checkpoint tuple: {e}")
            raise DatabaseError(f"Checkpoint retrieval failed: {e}") from e
    
    @log_function_call(logger)
    async def aput_tuple(
        self, 
        config: RunnableConfig, 
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata
    ) -> RunnableConfig:
        """
        Store a checkpoint in the database.
        
        Args:
            config: Runnable configuration
            checkpoint: Checkpoint data to store
            metadata: Checkpoint metadata
            
        Returns:
            Updated RunnableConfig with checkpoint information
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If inputs are invalid
        """
        try:
            await self._ensure_tables_exist()
            
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                raise ValidationError("thread_id is required in config.configurable")
            
            # Extract timestamp from checkpoint
            checkpoint_ts = datetime.fromisoformat(checkpoint["ts"])
            parent_ts = None
            if "parent" in checkpoint and "ts" in checkpoint["parent"]:
                parent_ts = datetime.fromisoformat(checkpoint["parent"]["ts"])
            
            # Prepare checkpoint data (exclude system fields)
            checkpoint_data = {k: v for k, v in checkpoint.items() 
                             if k not in ["v", "id", "ts"]}
            
            # Store checkpoint
            query = """
            INSERT INTO langgraph_checkpoints 
            (thread_id, thread_ts, parent_ts, checkpoint, metadata)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (thread_id, thread_ts) 
            DO UPDATE SET 
                checkpoint = EXCLUDED.checkpoint,
                metadata = EXCLUDED.metadata
            """
            
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    query,
                    thread_id,
                    checkpoint_ts, 
                    parent_ts,
                    json.dumps(checkpoint_data),
                    json.dumps(metadata or {})
                )
                await conn.commit()
            
            logger.debug(f"Stored checkpoint for thread_id: {thread_id}")
            
            # Return updated config
            updated_config = config.copy()
            updated_config.setdefault("configurable", {})["thread_ts"] = checkpoint["ts"]
            
            return updated_config
            
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            raise DatabaseError(f"Checkpoint storage failed: {e}") from e
    
    @log_function_call(logger)
    async def alist_tuples(
        self, 
        config: RunnableConfig,
        before: Optional[str] = None,
        limit: Optional[int] = None
    ) -> AsyncIterator[Tuple[str, Checkpoint]]:
        """
        List checkpoints for a thread with optional filtering.
        
        Args:
            config: Runnable configuration containing thread information
            before: Optional timestamp to filter checkpoints before
            limit: Optional limit on number of results
            
        Yields:
            Tuples of (checkpoint_id, checkpoint)
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If config is invalid
        """
        try:
            await self._ensure_tables_exist()
            
            thread_id = config.get("configurable", {}).get("thread_id")
            if not thread_id:
                raise ValidationError("thread_id is required in config.configurable")
            
            # Build query with optional filters
            query_parts = [
                "SELECT thread_ts, checkpoint, metadata",
                "FROM langgraph_checkpoints",
                "WHERE thread_id = $1"
            ]
            params = [thread_id]
            
            if before:
                query_parts.append("AND thread_ts < $2")
                params.append(datetime.fromisoformat(before))
            
            query_parts.append("ORDER BY thread_ts DESC")
            
            if limit:
                query_parts.append(f"LIMIT ${len(params) + 1}")
                params.append(limit)
            
            query = " ".join(query_parts)
            
            async with self.db_manager.get_connection() as conn:
                async for record in conn.cursor(query, *params):
                    checkpoint_data = {
                        "v": 1,
                        "id": record["thread_ts"].isoformat(),
                        "ts": record["thread_ts"].isoformat(),
                        **json.loads(record["checkpoint"])
                    }
                    
                    checkpoint = Checkpoint(**checkpoint_data)
                    yield (record["thread_ts"].isoformat(), checkpoint)
            
            logger.debug(f"Listed checkpoints for thread_id: {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise DatabaseError(f"Checkpoint listing failed: {e}") from e
    
    @log_function_call(logger)
    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: List[Tuple[str, Any]],
        task_id: str
    ) -> None:
        """
        Store write operations for a checkpoint.
        
        Args:
            config: Runnable configuration
            writes: List of (channel, value) tuples to store
            task_id: Task identifier for the writes
            
        Raises:
            DatabaseError: If database operation fails
            ValidationError: If inputs are invalid
        """
        try:
            await self._ensure_tables_exist()
            
            thread_id = config.get("configurable", {}).get("thread_id")
            thread_ts = config.get("configurable", {}).get("thread_ts")
            
            if not thread_id:
                raise ValidationError("thread_id is required in config.configurable")
            if not thread_ts:
                raise ValidationError("thread_ts is required in config.configurable")
            
            if not writes:
                logger.debug("No writes to store")
                return
            
            # Convert thread_ts to datetime
            checkpoint_ts = datetime.fromisoformat(thread_ts)
            
            # Prepare write records
            records = []
            for idx, (channel, value) in enumerate(writes):
                records.append((
                    thread_id,
                    checkpoint_ts,
                    task_id,
                    idx,
                    channel,
                    json.dumps(value)
                ))
            
            # Store writes
            query = """
            INSERT INTO langgraph_writes 
            (thread_id, thread_ts, task_id, idx, channel, value)
            VALUES ($1, $2, $3, $4, $5, $6)
            """
            
            async with self.db_manager.get_connection() as conn:
                await conn.executemany(query, records)
                await conn.commit()
            
            logger.debug(f"Stored {len(writes)} writes for thread_id: {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to store writes: {e}")
            raise DatabaseError(f"Write storage failed: {e}") from e


# === Factory Functions ===

@log_function_call(logger)
async def create_postgresql_checkpointer(
    db_manager: Optional[DatabaseManager] = None
) -> PostgreSQLCheckpointer:
    """
    Factory function to create and initialize PostgreSQL checkpointer.
    
    Args:
        db_manager: Optional DatabaseManager instance
        
    Returns:
        PostgreSQLCheckpointer: Initialized checkpointer
        
    Raises:
        DatabaseError: If initialization fails
    """
    try:
        checkpointer = PostgreSQLCheckpointer(db_manager)
        await checkpointer._ensure_tables_exist()
        
        logger.info("PostgreSQL checkpointer created and initialized")
        return checkpointer
        
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
        raise DatabaseError(f"Checkpointer creation failed: {e}") from e


@log_function_call(logger)
def create_memory_checkpointer() -> "MemoryCheckpointer":
    """
    Create an in-memory checkpointer for testing and development.
    
    Returns:
        MemoryCheckpointer: In-memory checkpointer instance
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
        logger.info("In-memory checkpointer created for development/testing")
        return checkpointer
        
    except ImportError:
        logger.warning("MemorySaver not available, using None")
        return None
