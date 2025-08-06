import sys
import logging
from typing import Dict, Any, List
from ..core.config import Settings
from ..core.exceptions import DependencyError, ConfigurationError

logger = logging.getLogger(__name__)


class DependencyValidator:
    """Professional dependency validator."""
    
    @staticmethod
    def validate_langchain_dependencies() -> None:
        """Validate LangChain dependencies are available."""
        required_modules = [
            "langchain_google_genai",
            "langchain_core.prompts", 
            "langchain_core.output_parsers",
            "langchain_core.runnables"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            raise DependencyError(
                f"Missing required LangChain modules: {missing_modules}",
                dependency_name="langchain"
            )
    
    @staticmethod
    def validate_gemini_configuration(settings: Settings) -> None:
        """Validate Gemini configuration is complete."""
        gemini_config = settings.gemini_config
        required_fields = ["model_name", "api_key"]
        
        missing_fields = [
            field for field in required_fields 
            if not gemini_config.get(field)
        ]
        
        if missing_fields:
            raise ConfigurationError(
                f"Missing required Gemini configuration: {missing_fields}",
                config_key="gemini_config"
            )
    
    @staticmethod
    def validate_all_dependencies(settings: Settings) -> Dict[str, str]:
        """
        Validate all application dependencies.
        
        Returns:
            Dict[str, str]: Status of each dependency
            
        Raises:
            DependencyError: If critical dependencies are missing
        """
        status = {}
        
        try:
            DependencyValidator.validate_langchain_dependencies()
            status["langchain"] = "OK"
        except DependencyError as e:
            logger.error(f"LangChain validation failed: {e}")
            status["langchain"] = f"FAILED: {e.message}"
            raise
        
        try:
            DependencyValidator.validate_gemini_configuration(settings)
            status["gemini_config"] = "OK"
        except ConfigurationError as e:
            logger.error(f"Gemini configuration validation failed: {e}")
            status["gemini_config"] = f"FAILED: {e.message}"
            raise
        
        return status


def perform_startup_validation(settings: Settings) -> None:
    """
    Perform startup validation - fail fast if dependencies missing.
    
    Args:
        settings: Application settings
        
    Raises:
        DependencyError: If validation fails
    """
    logger.info("Starting dependency validation...")
    
    try:
        status = DependencyValidator.validate_all_dependencies(settings)
        logger.info(f"All dependencies validated successfully: {status}")
    except (DependencyError, ConfigurationError) as e:
        logger.critical(f"Dependency validation failed: {e}")
        logger.critical("Application cannot start with missing dependencies")
        sys.exit(1)  # Fail fast - don't use fallbacks
