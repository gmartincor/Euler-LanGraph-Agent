"""Professional Test Infrastructure - Mock Factory for API Testing.

This module provides a centralized mock factory to eliminate real API calls
in tests, following DRY, KISS, and professional testing patterns.

Key Features:
- Centralized Mock Management: Single source of truth for all mocks
- Zero API Calls: Prevents real API consumption in tests
- DRY Principle: Reusable mock configurations
- Professional Standards: Consistent mock behavior across all tests
"""

from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch
import pytest
from contextlib import contextmanager

from app.core.config import Settings
from app.tools.registry import ToolRegistry


class MockFactory:
    """Centralized factory for creating consistent mocks across all tests."""
    
    @staticmethod
    def create_mock_settings(override_config: Optional[Dict[str, Any]] = None) -> Mock:
        """
        Create mock settings that never use real API keys.
        
        Args:
            override_config: Optional config overrides for specific tests
            
        Returns:
            Mock settings object with safe test configuration
        """
        default_config = {
            # Safe test configuration - never real API keys
            "google_api_key": "mock_api_key_12345",
            "gemini_api_key": "mock_api_key_12345",  # Consistent with test expectations
            "gemini_model_name": "gemini-1.5-pro",
            "gemini_temperature": 0.1,
            "gemini_max_tokens": 8192,
            "gemini_top_p": 0.9,
            "gemini_top_k": 40,
            "database_url": "sqlite:///:memory:",
            "debug": True,
            "environment": "testing"
        }
        
        if override_config:
            default_config.update(override_config)
        
        # Create mock settings with safe configuration
        mock_settings = Mock(spec=Settings)
        
        # Add individual attributes
        for key, value in default_config.items():
            setattr(mock_settings, key, value)
        
        # Add gemini_config property
        mock_settings.gemini_config = {
            "model_name": default_config["gemini_model_name"],
            "api_key": default_config["google_api_key"],
            "temperature": default_config["gemini_temperature"],
            "max_tokens": default_config["gemini_max_tokens"],
            "top_p": default_config["gemini_top_p"],
            "top_k": default_config["gemini_top_k"],
        }
        
        return mock_settings
    
    @staticmethod
    def create_mock_tool_registry() -> Mock:
        """Create mock tool registry for testing."""
        mock_registry = Mock(spec=ToolRegistry)
        mock_registry.list_tools.return_value = ["integral_tool", "plot_tool", "analysis_tool"]
        mock_registry.get_tool.return_value = Mock(
            name="test_tool",
            description="Mock tool for testing",
            usage_stats={"success_rate": 0.95, "avg_execution_time": 0.1}
        )
        return mock_registry
    
    @staticmethod
    def create_mock_llm() -> Mock:
        """Create mock LLM that never makes real API calls and returns proper responses."""
        mock_llm = Mock()
        
        # Mock LLM responses should return JSON-like strings for chains
        mock_llm.ainvoke = AsyncMock(return_value='{"analysis": "mock", "confidence": 0.8}')
        mock_llm.invoke = Mock(return_value='{"analysis": "mock", "confidence": 0.8}')
        
        # For chains that expect a simple string response
        mock_llm.stream = AsyncMock()
        
        return mock_llm
    
    @staticmethod
    def create_mock_chain_responses() -> Dict[str, Dict[str, Any]]:
        """
        Create consistent mock responses for all chain types.
        
        Returns:
            Dictionary with mock responses for each chain type
        """
        return {
            "analysis": {
                "problem_type": "integral",
                "complexity": "medium",
                "requires_tools": True,
                "description": "Mock analysis result",
                "approach": "Mock mathematical approach",
                "confidence": 0.85
            },
            "reasoning": {
                "approach": "Mock reasoning approach",
                "steps": ["Step 1: Mock step", "Step 2: Mock step"],
                "tools_needed": ["integral_tool"],
                "confidence": 0.90
            },
            "validation": {
                "is_valid": True,
                "score": 0.92,
                "issues": [],
                "feedback": "Mock validation passed"
            },
            "response": {
                "answer": "42",
                "steps": ["Mock step 1", "Mock step 2", "Mock step 3"],
                "explanation": "Mock mathematical explanation",
                "confidence": 0.88
            }
        }
    
    @staticmethod
    def create_mock_chain_factory(chain_responses: Optional[Dict[str, Dict[str, Any]]] = None) -> Mock:
        """
        Create mock chain factory with consistent responses.
        
        Args:
            chain_responses: Optional custom responses for chains
            
        Returns:
            Mock chain factory with all chain methods mocked
        """
        if chain_responses is None:
            chain_responses = MockFactory.create_mock_chain_responses()
        
        mock_factory = Mock()
        
        # Create mock chains for each type
        for chain_type, response in chain_responses.items():
            mock_chain = AsyncMock()
            mock_chain.ainvoke.return_value = response
            
            # Add chain to factory
            setattr(mock_factory, f"create_{chain_type}_chain", Mock(return_value=mock_chain))
        
        return mock_factory
    
    @staticmethod
    @contextmanager
    def mock_all_api_calls():
        """
        Context manager to mock ALL API-related calls in tests.
        
        This is the master mock that prevents any real API calls.
        """
        with patch('app.core.config.get_settings') as mock_get_settings, \
             patch('app.agents.nodes.get_settings') as mock_nodes_get_settings, \
             patch('app.agents.chains.ChatGoogleGenerativeAI') as mock_llm_class, \
             patch('app.agents.nodes.create_bigtool_manager') as mock_bigtool, \
             patch('app.agents.nodes._get_chain_factory') as mock_get_chain_factory:
            
            # Mock settings everywhere
            mock_settings = MockFactory.create_mock_settings()
            mock_get_settings.return_value = mock_settings
            mock_nodes_get_settings.return_value = mock_settings
            
            # Mock LLM class
            mock_llm_class.return_value = MockFactory.create_mock_llm()
            
            # Mock BigTool manager
            mock_bigtool_manager = Mock()
            mock_bigtool_manager.search_tools.return_value = []
            mock_bigtool.return_value = mock_bigtool_manager
            
            # Mock chain factory
            mock_get_chain_factory.return_value = MockFactory.create_mock_chain_factory()
            
            yield {
                'settings': mock_settings,
                'llm_class': mock_llm_class,
                'bigtool': mock_bigtool,
                'chain_factory': mock_get_chain_factory
            }


# Pytest fixtures for easy reuse across all tests
@pytest.fixture
def mock_settings():
    """Pytest fixture for mock settings."""
    return MockFactory.create_mock_settings()


@pytest.fixture
def mock_tool_registry():
    """Pytest fixture for mock tool registry."""
    return MockFactory.create_mock_tool_registry()


@pytest.fixture
def mock_chain_factory():
    """Pytest fixture for mock chain factory."""
    return MockFactory.create_mock_chain_factory()


@pytest.fixture
def mock_all_apis():
    """Pytest fixture to mock all API calls."""
    with MockFactory.mock_all_api_calls() as mocks:
        yield mocks


# Test validation helpers
class TestValidationHelpers:
    """Helper methods for test validation."""
    
    @staticmethod
    def assert_no_real_api_calls(mock_llm_class):
        """Assert that no real API calls were made."""
        # Verify LLM class was called with test API key only
        if mock_llm_class.called:
            call_args = mock_llm_class.call_args
            if call_args and 'api_key' in call_args.kwargs:
                api_key = call_args.kwargs['api_key']
                # Professional pattern: Accept both test prefixes for consistency
                assert (api_key.startswith('test-') or api_key.startswith('mock_')), f"Real API key used: {api_key[:10]}..."
    
    @staticmethod
    def assert_valid_mock_response(result: Dict[str, Any]):
        """Validate that a result came from mocks, not real API."""
        # Check for mock indicators
        if 'problem_analysis' in result:
            description = result['problem_analysis'].get('description', '')
            assert 'Mock' in description, "Result doesn't appear to be from mock"
        
        # Confidence scores should be in mock ranges
        confidence = result.get('confidence_score', 0)
        assert 0.8 <= confidence <= 0.95, f"Confidence {confidence} outside mock range"
