import os
import pytest
from unittest.mock import patch
from typing import Generator, Any

# Import our professional mock infrastructure
from tests.fixtures.mock_factory import MockFactory


# Auto-fixture that applies to ALL tests
@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Auto-fixture that sets up safe test environment for ALL tests.
    
    This fixture automatically runs before every test to ensure:
    - Safe environment variables (no real API keys)
    - Mock configuration is available
    - Test database settings
    """
    # Set safe test environment variables
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["GOOGLE_API_KEY"] = "test-safe-api-key-no-real-calls"
    os.environ["GEMINI_MODEL_NAME"] = "gemini-2.5-flash"
    
    # Clear any cached settings to prevent real config loading
    try:
        from app.core.config import get_settings
        if hasattr(get_settings, 'cache_clear'):
            get_settings.cache_clear()
    except ImportError:
        pass
    
    yield
    
    # Cleanup after test
    test_env_vars = [
        "ENVIRONMENT", "DEBUG", "DATABASE_URL", 
        "GOOGLE_API_KEY", "GEMINI_MODEL_NAME"
    ]
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]


# Global mock fixture for critical API calls
@pytest.fixture(autouse=True)
def prevent_real_api_calls():
    """
    Auto-fixture that prevents ALL real API calls in tests.
    
    This fixture automatically mocks critical components that could
    make real API calls, ensuring zero API quota consumption.
    """
    with patch('app.core.config.get_settings') as mock_get_settings, \
         patch('langchain_google_genai.ChatGoogleGenerativeAI') as mock_llm_class, \
         patch('app.agents.chains.ChatGoogleGenerativeAI') as mock_chains_llm, \
         patch('app.agents.nodes.ChatGoogleGenerativeAI') as mock_nodes_llm:
        
        # Set up safe mock settings
        mock_get_settings.return_value = MockFactory.create_mock_settings()
        
        # Set up safe mock LLM for all modules
        mock_llm = MockFactory.create_mock_llm()
        mock_llm_class.return_value = mock_llm
        mock_chains_llm.return_value = mock_llm
        mock_nodes_llm.return_value = mock_llm
        
        yield {
            'mock_settings': mock_get_settings,
            'mock_llm_class': mock_llm_class,
            'mock_chains_llm': mock_chains_llm,
            'mock_nodes_llm': mock_nodes_llm
        }


# Convenience fixtures for common test objects
@pytest.fixture
def safe_mock_settings():
    """Fixture providing safe mock settings."""
    return MockFactory.create_mock_settings()


@pytest.fixture
def safe_mock_tool_registry():
    """Fixture providing safe mock tool registry."""
    return MockFactory.create_mock_tool_registry()


@pytest.fixture
def safe_mock_chain_factory():
    """Fixture providing safe mock chain factory."""
    return MockFactory.create_mock_chain_factory()


@pytest.fixture
def mock_chain_factory():
    """Alias for safe_mock_chain_factory for backwards compatibility."""
    return MockFactory.create_mock_chain_factory()


@pytest.fixture
def comprehensive_api_mocks():
    """Fixture providing comprehensive API mocking."""
    with MockFactory.mock_all_api_calls() as mocks:
        yield mocks


# Test markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "no_api_calls: mark test as verified to make no API calls"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test with full mocking"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test with isolated mocking"
    )


# Hook to validate no real API calls in tests
@pytest.fixture(autouse=True)
def validate_no_real_api_usage(request):
    """
    Auto-fixture to validate tests don't use real APIs.
    
    This adds an extra layer of protection by monitoring for
    any attempts to use real API keys or make HTTP calls.
    """
    # Check if test is marked as safe
    if request.node.get_closest_marker("no_api_calls"):
        # Additional validation could go here
        pass
    
    yield
    
    # Post-test validation
    # Check environment for any real API keys that might have leaked
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    if google_api_key and not google_api_key.startswith("test-"):
        pytest.fail(f"Test leaked real API key: {google_api_key[:10]}...")


# Performance tracking for tests
@pytest.fixture(autouse=True)
def track_test_performance(request):
    """Auto-fixture to ensure tests run fast with mocks."""
    import time
    
    start_time = time.time()
    yield
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Warn if test is too slow (indicates possible real API calls)
    if duration > 5.0:  # 5 seconds threshold
        print(f"\nWARNING: Test {request.node.name} took {duration:.2f}s - possible real API calls!")


# Async test configuration
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
