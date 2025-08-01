"""Unit tests for core configuration."""

import os
import pytest
from unittest.mock import patch

from app.core.config import Settings, get_settings


class TestSettings:
    """Test the Settings class."""
    
    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            google_api_key="test-key"
        )
        
        assert settings.app_name == "ReAct Integral Agent"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert settings.environment == "development"
        assert settings.gemini_model_name == "gemini-1.5-pro"
        assert settings.gemini_temperature == 0.1
        assert settings.log_level == "INFO"
    
    def test_environment_validation(self):
        """Test environment validation."""
        with pytest.raises(ValueError, match="Environment must be one of"):
            Settings(
                database_url="postgresql://test:test@localhost/test",
                google_api_key="test-key",
                environment="invalid"
            )
    
    def test_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValueError, match="Temperature must be between 0.0 and 1.0"):
            Settings(
                database_url="postgresql://test:test@localhost/test",
                google_api_key="test-key",
                gemini_temperature=1.5
            )
    
    def test_log_level_validation(self):
        """Test log level validation."""
        with pytest.raises(ValueError, match="Log level must be one of"):
            Settings(
                database_url="postgresql://test:test@localhost/test",
                google_api_key="test-key",
                log_level="INVALID"
            )
    
    def test_is_development_property(self):
        """Test is_development property."""
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            google_api_key="test-key",
            environment="development"
        )
        assert settings.is_development is True
        
        settings.environment = "production"
        assert settings.is_development is False
    
    def test_is_production_property(self):
        """Test is_production property."""
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            google_api_key="test-key",
            environment="production"
        )
        assert settings.is_production is True
        
        settings.environment = "development"
        assert settings.is_production is False
    
    def test_database_config_property(self):
        """Test database_config property."""
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            google_api_key="test-key",
            database_pool_size=10,
            database_max_overflow=20,
            debug=True,
            environment="development"
        )
        
        config = settings.database_config
        assert config["url"] == "postgresql://test:test@localhost/test"
        assert config["pool_size"] == 10
        assert config["max_overflow"] == 20
        assert config["echo"] is True  # debug=True and development
    
    def test_gemini_config_property(self):
        """Test gemini_config property."""
        settings = Settings(
            database_url="postgresql://test:test@localhost/test",
            google_api_key="test-api-key",
            gemini_model_name="gemini-1.5-pro",
            gemini_temperature=0.5,
            gemini_max_tokens=2048,
            gemini_top_p=0.8,
            gemini_top_k=30
        )
        
        config = settings.gemini_config
        assert config["api_key"] == "test-api-key"
        assert config["model_name"] == "gemini-1.5-pro"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 2048
        assert config["top_p"] == 0.8
        assert config["top_k"] == 30


class TestGetSettings:
    """Test the get_settings function."""
    
    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://test:test@localhost/test",
        "GOOGLE_API_KEY": "test-key",
        "DEBUG": "true",
        "ENVIRONMENT": "testing"
    })
    def test_get_settings_with_env_vars(self):
        """Test get_settings with environment variables."""
        # Clear the cache
        get_settings.cache_clear()
        
        settings = get_settings()
        assert settings.database_url == "postgresql://test:test@localhost/test"
        assert settings.google_api_key == "test-key"
        assert settings.debug is True
        assert settings.environment == "testing"
    
    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
