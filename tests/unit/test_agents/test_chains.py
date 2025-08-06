"""Unit tests for chain factory - Professional Mock Infrastructure.

Tests for chain factory using centralized mock infrastructure to prevent
ALL real API calls and eliminate API quota consumption.

Key Principles:
- Zero API Calls: Complete mock infrastructure
- Professional Standards: Centralized mock management
- Fast Execution: Mock responses in microseconds
- Cost Effective: Zero API quota usage
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import our professional mock infrastructure
from tests.fixtures.mock_factory import MockFactory, TestValidationHelpers

# Test with error handling for optional dependencies
try:
    from app.agents.chains import ChainFactory, create_chain_factory, create_all_chains
    CHAINS_AVAILABLE = True
except ImportError:
    CHAINS_AVAILABLE = False


@pytest.mark.skipif(not CHAINS_AVAILABLE, reason="Chain dependencies not available")
class TestChainFactory:
    """Test cases for ChainFactory with ZERO API calls."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create MOCK settings - never uses real API keys."""
        return MockFactory.create_mock_settings()
    
    @pytest.fixture  
    def mock_tool_registry(self):
        """Create MOCK tool registry."""
        return MockFactory.create_mock_tool_registry()
    
    def test_chain_factory_initialization(self, mock_settings, mock_tool_registry):
        """Test ChainFactory initialization with MOCKED dependencies."""
        with patch('app.agents.chains.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm_class.return_value = MockFactory.create_mock_llm()
            
            factory = ChainFactory(mock_settings, mock_tool_registry)
            
            assert factory.settings == mock_settings
            assert factory.tool_registry == mock_tool_registry
            assert factory.llm is not None
            
            # Validate NO real API calls
            TestValidationHelpers.assert_no_real_api_calls(mock_llm_class)
    
    def test_chain_factory_with_provided_llm(self, mock_settings, mock_tool_registry):
        """Test ChainFactory initialization with provided MOCK LLM."""
        mock_llm = MockFactory.create_mock_llm()
        
        factory = ChainFactory(mock_settings, mock_tool_registry, mock_llm)
        
        assert factory.settings == mock_settings
        assert factory.tool_registry == mock_tool_registry
        assert factory.llm == mock_llm
    
    @patch('app.agents.chains.ChatGoogleGenerativeAI')
    def test_create_llm_never_uses_real_api_key(self, mock_llm_class, mock_settings, mock_tool_registry):
        """Test LLM creation NEVER uses real API keys.""" 
        mock_llm_class.return_value = MockFactory.create_mock_llm()
        
        factory = ChainFactory(mock_settings, mock_tool_registry)
        
        # Verify LLM was created with MOCK configuration only
        mock_llm_class.assert_called_once()
        call_kwargs = mock_llm_class.call_args.kwargs
        
        # Ensure API key is from our mock
        assert call_kwargs['api_key'] == "mock_api_key_12345"  # Fix: Use consistent mock API key
        assert call_kwargs['model'] == "gemini-2.5-flash"
        assert call_kwargs['temperature'] == 0.1
        
        # Validate NO real API calls
        TestValidationHelpers.assert_no_real_api_calls(mock_llm_class)
    
    def test_create_reasoning_chain(self, mock_settings, mock_tool_registry):
        """Test reasoning chain creation."""
        factory = ChainFactory(mock_settings, mock_tool_registry)
        
        result = factory.create_reasoning_chain()
        
        # Verify chain was created and is callable
        assert result is not None
        assert hasattr(result, 'invoke') or hasattr(result, '__call__')
        
        # Verify it's a LangChain runnable
        from langchain_core.runnables import Runnable
        assert isinstance(result, Runnable)
    
    def test_create_tool_selection_chain(self, mock_settings, mock_tool_registry):
        """Test tool selection chain creation."""
        factory = ChainFactory(mock_settings, mock_tool_registry)
        
        result = factory.create_tool_selection_chain()
        
        # Verify chain was created and is callable
        assert result is not None
        assert hasattr(result, 'invoke') or hasattr(result, '__call__')
    
    def test_create_validation_chain(self, mock_settings, mock_tool_registry):
        """Test validation chain creation."""
        factory = ChainFactory(mock_settings, mock_tool_registry)
        
        result = factory.create_validation_chain()
        
        # Verify chain was created and is callable
        assert result is not None
        assert hasattr(result, 'invoke') or hasattr(result, '__call__')


@pytest.mark.skipif(not CHAINS_AVAILABLE, reason="Chain dependencies not available")
class TestChainFactoryFunctions:
    """Test factory functions with ZERO API calls."""
    
    def test_create_chain_factory_never_uses_real_api(self):
        """Test create_chain_factory with MOCK dependencies only."""
        mock_settings = MockFactory.create_mock_settings()
        mock_tool_registry = MockFactory.create_mock_tool_registry()
        
        with patch('app.agents.chains.ChatGoogleGenerativeAI') as mock_llm_class:
            mock_llm_class.return_value = MockFactory.create_mock_llm()
            
            factory = create_chain_factory(mock_settings, mock_tool_registry)
            
            assert isinstance(factory, ChainFactory)  
            assert factory.settings == mock_settings
            assert factory.tool_registry == mock_tool_registry
            
            # Validate NO real API calls
            TestValidationHelpers.assert_no_real_api_calls(mock_llm_class)
    
    def test_create_chain_factory_with_mock_llm(self):
        """Test create_chain_factory with provided MOCK LLM."""
        mock_settings = MockFactory.create_mock_settings()
        mock_tool_registry = MockFactory.create_mock_tool_registry()
        mock_llm = MockFactory.create_mock_llm()
        
        factory = create_chain_factory(mock_settings, mock_tool_registry, mock_llm)
        
        assert isinstance(factory, ChainFactory)
        assert factory.llm == mock_llm


# Simple integration test that doesn't require external dependencies
def test_chain_module_imports():
    """Test that chain module can be imported successfully."""
    try:
        from app.agents import chains
        assert hasattr(chains, 'ChainFactory')
        assert hasattr(chains, 'create_chain_factory')
        assert hasattr(chains, 'create_all_chains')
    except ImportError:
        pytest.skip("Chain module dependencies not available")
