"""Configuration management for the ReAct Agent application."""

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="ReAct Integral Agent", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")
    
    # Gemini AI
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", env="GEMINI_MODEL")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    gemini_max_tokens: int = Field(default=1024, env="GEMINI_MAX_TOKENS")
    
    # BigTool
    bigtool_cache_ttl: int = Field(default=3600, env="BIGTOOL_CACHE_TTL")
    bigtool_max_tools: int = Field(default=10, env="BIGTOOL_MAX_TOOLS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    
    # Streamlit
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Security
    session_secret: str = Field(default="dev-secret-key", env="SESSION_SECRET")
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @validator("gemini_temperature")
    def validate_temperature(cls, v: float) -> float:
        """Validate Gemini temperature."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
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
        """Get Gemini AI configuration dictionary."""
        return {
            "api_key": self.google_api_key,
            "model": self.gemini_model,
            "temperature": self.gemini_temperature,
            "max_tokens": self.gemini_max_tokens,
        }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
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
