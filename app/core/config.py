
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="ReAct Integral Agent", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")  # Single debug field, defaults to True for development
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    
    # Gemini AI (Updated for Gemini 2.5 Flash - Better rate limits)
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model_name: str = Field(default="gemini-2.5-flash", env="GEMINI_MODEL_NAME")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=8192, env="GEMINI_MAX_TOKENS")  # 2.5 Flash supports up to 8192
    gemini_top_p: float = Field(default=0.9, env="GEMINI_TOP_P")
    gemini_top_k: int = Field(default=40, env="GEMINI_TOP_K")
    
    # LangGraph Configuration
    langgraph_checkpointer_type: str = Field(default="postgresql", env="LANGGRAPH_CHECKPOINTER_TYPE")
    langgraph_max_concurrent_sessions: int = Field(default=10, env="LANGGRAPH_MAX_CONCURRENT_SESSIONS")
    langgraph_session_timeout_minutes: int = Field(default=30, env="LANGGRAPH_SESSION_TIMEOUT_MINUTES")
    max_conversation_turns: int = Field(default=50, env="MAX_CONVERSATION_TURNS")
    agent_max_iterations: int = Field(default=10, env="AGENT_MAX_ITERATIONS")
    
    # BigTool Configuration
    bigtool_enabled: bool = Field(default=True, env="BIGTOOL_ENABLED")
    bigtool_max_tools: int = Field(default=50, env="BIGTOOL_MAX_TOOLS")
    bigtool_similarity_threshold: float = Field(default=0.7, env="BIGTOOL_SIMILARITY_THRESHOLD")
    bigtool_index_batch_size: int = Field(default=100, env="BIGTOOL_INDEX_BATCH_SIZE")
    bigtool_search_limit: int = Field(default=5, env="BIGTOOL_SEARCH_LIMIT")
    bigtool_vector_store: str = Field(default="in_memory", env="BIGTOOL_VECTOR_STORE")
    bigtool_embedding_model: str = Field(default="text-embedding-ada-002", env="BIGTOOL_EMBEDDING_MODEL")
    tool_search_top_k: int = Field(default=3, env="TOOL_SEARCH_TOP_K")
    memory_store_size: int = Field(default=1000, env="MEMORY_STORE_SIZE")
    bigtool_cache_ttl: int = Field(default=3600, env="BIGTOOL_CACHE_TTL")
    
    # Cache Configuration
    calculation_cache_ttl_hours: int = Field(default=24, env="CALCULATION_CACHE_TTL_HOURS")
    conversation_history_limit: int = Field(default=100, env="CONVERSATION_HISTORY_LIMIT")
    
    # Redis Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Streamlit
    streamlit_server_address: str = Field(default="0.0.0.0", env="STREAMLIT_SERVER_ADDRESS")
    streamlit_server_port: int = Field(default=8501, env="STREAMLIT_SERVER_PORT")
    # Additional streamlit fields for compatibility
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Security
    session_secret: str = Field(default="dev-secret-key", env="SESSION_SECRET")
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @field_validator("gemini_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate Gemini temperature."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    @field_validator("gemini_max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate Gemini max tokens for 2.5 Flash model."""
        if v < 1 or v > 8192:
            raise ValueError("Gemini 2.5 Flash max tokens must be between 1 and 8192")
        return v
    
    @field_validator("gemini_top_p")
    @classmethod
    def validate_top_p(cls, v: float) -> float:
        """Validate Gemini top_p."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
        return v
    
    @field_validator("agent_max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max iterations."""
        if v < 1 or v > 50:
            raise ValueError("Max iterations must be between 1 and 50")
        return v
    
    @field_validator("tool_search_top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k for tool search."""
        if v < 1 or v > 10:
            raise ValueError("Top_k must be between 1 and 10")
        return v
    
    @field_validator("gemini_top_k")
    @classmethod
    def validate_gemini_top_k(cls, v: int) -> int:
        """Validate Gemini top_k."""
        if v < 1 or v > 100:
            raise ValueError("Gemini top_k must be between 1 and 100")
        return v
    
    @field_validator("bigtool_similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v: float) -> float:
        """Validate BigTool similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("langgraph_checkpointer_type")
    @classmethod
    def validate_checkpointer_type(cls, v: str) -> str:
        """Validate LangGraph checkpointer type."""
        valid_types = ["postgresql", "sqlite", "memory"]
        if v.lower() not in valid_types:
            raise ValueError(f"Checkpointer type must be one of: {valid_types}")
        return v.lower()
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration dictionary."""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug and self.is_development,
        }
    
    @property
    def gemini_config(self) -> Dict[str, Any]:
        """Get Gemini AI configuration dictionary optimized for 2.5 Flash."""
        return {
            "api_key": self.google_api_key,
            "model_name": self.gemini_model_name,
            "temperature": self.gemini_temperature,
            "max_output_tokens": self.gemini_max_tokens,  # Correct parameter for Google Gemini
            "top_p": self.gemini_top_p,
            "top_k": self.gemini_top_k,
            "embedding_model": "text-embedding-004",  # Google's latest text embedding model
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
    
    @property
    def langgraph_config(self) -> Dict[str, Any]:
        """Get LangGraph configuration dictionary."""
        return {
            "checkpointer_type": self.langgraph_checkpointer_type,
            "max_concurrent_sessions": self.langgraph_max_concurrent_sessions,
            "session_timeout_minutes": self.langgraph_session_timeout_minutes,
            "max_conversation_turns": self.max_conversation_turns,
            "max_iterations": self.agent_max_iterations,
        }
    
    @property
    def bigtool_config(self) -> Dict[str, Any]:
        """Get BigTool configuration dictionary."""
        return {
            "enabled": self.bigtool_enabled,
            "max_tools": self.bigtool_max_tools,
            "similarity_threshold": self.bigtool_similarity_threshold,
            "index_batch_size": self.bigtool_index_batch_size,
            "search_limit": self.bigtool_search_limit,
            "vector_store": self.bigtool_vector_store,
            "embedding_model": self.bigtool_embedding_model,
            "top_k": self.tool_search_top_k,
            "memory_size": self.memory_store_size,
            "cache_ttl": self.bigtool_cache_ttl,
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields instead of raising errors
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Returns:
        Settings: Application configuration instance
    """
    return Settings()


def get_database_url() -> str:
    """Get the database URL from settings."""
    return get_settings().database_url


def get_google_api_key() -> str:
    """Get the Google API key from settings."""
    return get_settings().google_api_key
