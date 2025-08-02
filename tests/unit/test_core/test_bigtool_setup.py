"""Unit tests for BigTool setup.

Simple tests to improve coverage following KISS principle.
Tests focus on basic functionality without external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from app.core.config import Settings
from app.core.exceptions import ToolError
from app.tools.registry import ToolRegistry

# Test with error handling for optional dependencies
try:
    from app.core.bigtool_setup import BigToolManager, create_bigtool_manager
    BIGTOOL_AVAILABLE = True
except ImportError:
    BIGTOOL_AVAILABLE = False


@pytest.mark.skipif(not BIGTOOL_AVAILABLE, reason="BigTool dependencies not available")
class TestBigToolManager:
    """Test cases for BigToolManager."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.bigtool_config = {
            "enabled": True,
            "max_tools": 50,
            "similarity_threshold": 0.7,
            "vector_store": "in_memory",
            "embedding_model": "text-embedding-004",
            "top_k": 3,
            "memory_size": 1000,
            "cache_ttl": 3600,
        }
        settings.gemini_config = {
            "model_name": "gemini-1.5-pro",
            "api_key": "test-key",
            "temperature": 0.1,
            "max_tokens": 8192,
        }
        return settings
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry with proper tool mocks."""
        registry = Mock(spec=ToolRegistry)
        
        # Create proper tool mocks with string properties
        integral_tool = Mock()
        integral_tool.name = "integral_tool"  # String, not Mock
        integral_tool.description = "Calculate definite and indefinite integrals"
        integral_tool.usage_stats = {"success_rate": 0.9}
        
        plot_tool = Mock()
        plot_tool.name = "plot_tool"  # String, not Mock
        plot_tool.description = "Create mathematical plots and visualizations"
        plot_tool.usage_stats = {"success_rate": 0.8}
        
        # Configure registry methods
        registry.list_tools.return_value = ["integral_tool", "plot_tool"]
        
        # Make get_tool return different tools based on name
        def mock_get_tool(name):
            if name == "integral_tool":
                return integral_tool
            elif name == "plot_tool":
                return plot_tool
            else:
                return integral_tool  # Default fallback
                
        registry.get_tool.side_effect = mock_get_tool
        
        return registry
    
    def test_bigtool_manager_initialization(self, mock_settings, mock_tool_registry):
        """Test BigToolManager initialization."""
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        assert manager.settings == mock_settings
        assert manager.tool_registry == mock_tool_registry
        assert not manager._is_initialized
        assert manager._agent is None
    
    def test_bigtool_manager_properties(self, mock_settings, mock_tool_registry):
        """Test BigToolManager properties."""
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        # Test is_enabled property
        assert manager.is_enabled is True
        
        # Test mock with disabled setting
        mock_settings.bigtool_config = {"enabled": False}
        manager_disabled = BigToolManager(mock_tool_registry, mock_settings)
        assert manager_disabled.is_enabled is False
    
    @patch('app.core.bigtool_setup.InMemoryStore')
    @patch('app.core.bigtool_setup.init_embeddings')
    @patch('app.core.bigtool_setup.init_chat_model')
    @patch('app.core.bigtool_setup.create_agent')
    async def test_initialize_success(self, mock_create_agent, mock_init_chat, 
                                    mock_init_embeddings, mock_in_memory_store, mock_settings, mock_tool_registry):
        """Test successful initialization."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_llm = Mock()
        mock_agent_builder = Mock()
        mock_compiled_agent = Mock()
        mock_store = Mock()
        
        mock_init_embeddings.return_value = mock_embeddings
        mock_init_chat.return_value = mock_llm
        mock_create_agent.return_value = mock_agent_builder
        mock_agent_builder.compile.return_value = mock_compiled_agent
        mock_in_memory_store.return_value = mock_store
        
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        # Test initialization
        await manager.initialize()
        
        assert manager._is_initialized is True
        assert manager._agent == mock_compiled_agent
        
        # Verify proper calls
        mock_init_embeddings.assert_called_once()
        mock_init_chat.assert_called_once()
        mock_create_agent.assert_called_once()
        mock_in_memory_store.assert_called_once()
        mock_store.put.assert_called()  # Verify tools were indexed
    
    @patch('app.core.bigtool_setup.init_embeddings')
    async def test_initialize_failure(self, mock_init_embeddings, mock_settings, mock_tool_registry):
        """Test initialization failure handling."""
        # Setup mock to raise exception
        mock_init_embeddings.side_effect = Exception("Test error")
        
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        # Test that initialization failure is handled
        with pytest.raises(Exception, match="Test error"):
            await manager.initialize()
        
        assert not manager._is_initialized
        assert manager._agent is None
    
    async def test_get_tool_recommendations_not_initialized(self, mock_settings, mock_tool_registry):
        """Test tool recommendations when not initialized."""
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        # Should raise ToolError when not initialized
        with pytest.raises(ToolError, match="BigTool not initialized"):
            await manager.get_tool_recommendations("test query")
    
    async def test_get_tool_recommendations_disabled(self, mock_settings, mock_tool_registry):
        """Test tool recommendations when disabled."""
        mock_settings.bigtool_config = {"enabled": False}
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        # Should raise ToolError when not initialized (disabled manager is never initialized)
        with pytest.raises(ToolError, match="BigTool not initialized"):
            await manager.get_tool_recommendations("test query")
    
    @patch('app.core.bigtool_setup.InMemoryStore')
    @patch('app.core.bigtool_setup.init_embeddings')
    @patch('app.core.bigtool_setup.init_chat_model')
    @patch('app.core.bigtool_setup.create_agent')
    async def test_search_tools_success(self, mock_create_agent, mock_init_chat, 
                                       mock_init_embeddings, mock_in_memory_store, mock_settings, mock_tool_registry):
        """Test successful tool search."""
        # Setup mocks
        mock_embeddings = Mock()
        mock_llm = Mock()
        mock_agent_builder = Mock()
        mock_compiled_agent = Mock()
        mock_store = Mock()
        
        # Mock store.search to return proper search results
        mock_search_result = Mock()
        mock_search_result.value = {"description": "integral_tool: Calculate definite and indefinite integrals"}
        mock_store.search.return_value = [mock_search_result]
        
        mock_compiled_agent.ainvoke = AsyncMock(return_value={
            "messages": [Mock(content="integral_tool,plot_tool")]
        })
        
        # Configure mock settings to avoid attribute errors
        mock_settings.tool_search_top_k = 3
        
        mock_init_embeddings.return_value = mock_embeddings
        mock_init_chat.return_value = mock_llm
        mock_create_agent.return_value = mock_agent_builder
        mock_agent_builder.compile.return_value = mock_compiled_agent
        mock_in_memory_store.return_value = mock_store
        
        manager = BigToolManager(mock_tool_registry, mock_settings)
        await manager.initialize()
        
        # Test search
        results = await manager.get_tool_recommendations("integral calculation", top_k=2)
        
        assert isinstance(results, list)
        # Mock should be called
        mock_compiled_agent.ainvoke.assert_called_once()
    
    def test_health_check_not_initialized(self, mock_settings, mock_tool_registry):
        """Test health check when not initialized."""
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        health = manager.health_check()
        
        assert health["status"] == "not_initialized"
        assert health["is_enabled"] is True
        assert health["is_initialized"] is False
    
    def test_health_check_disabled(self, mock_settings, mock_tool_registry):
        """Test health check when disabled."""
        mock_settings.bigtool_config = {"enabled": False}
        manager = BigToolManager(mock_tool_registry, mock_settings)
        
        health = manager.health_check()
        
        assert health["status"] == "disabled"
        assert health["is_enabled"] is False


@pytest.mark.skipif(not BIGTOOL_AVAILABLE, reason="BigTool dependencies not available")
class TestBigToolManagerFactory:
    """Test factory function for BigToolManager."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        settings = Mock(spec=Settings)
        settings.bigtool_config = {"enabled": True}
        settings.gemini_config = {
            "model_name": "gemini-1.5-pro",
            "temperature": 0.7,
            "max_tokens": 2048,
            "api_key": "test-api-key"
        }
        return settings
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry."""
        registry = Mock(spec=ToolRegistry)
        # Mock list_tools to return iterable tool names
        registry.list_tools.return_value = ["integral_tool", "plot_tool", "analysis_tool"]
        # Mock get_tool to return tool objects with proper properties
        def mock_get_tool(name):
            tool = Mock()
            tool.name = name
            tool.description = f"Description for {name}"
            return tool
        registry.get_tool.side_effect = mock_get_tool
        return registry
    
    @patch('app.core.bigtool_setup.InMemoryStore')
    @patch('app.core.bigtool_setup.init_embeddings')
    @patch('app.core.bigtool_setup.create_agent')
    @patch('app.core.bigtool_setup.init_chat_model')
    async def test_create_bigtool_manager(self, mock_init_chat, mock_create_agent, 
                                        mock_init_embeddings, mock_memory_store,
                                        mock_settings, mock_tool_registry):
        """Test create_bigtool_manager factory function."""
        # Setup mocks for initialization
        mock_store = Mock()
        mock_memory_store.return_value = mock_store
        mock_init_embeddings.return_value = Mock()
        mock_llm = Mock()
        mock_init_chat.return_value = mock_llm
        mock_builder = Mock()
        mock_create_agent.return_value = mock_builder
        mock_agent = Mock()
        mock_builder.compile.return_value = mock_agent
        
        manager = await create_bigtool_manager(mock_tool_registry, mock_settings)
        
        assert isinstance(manager, BigToolManager)
        assert manager.settings == mock_settings
        assert manager.tool_registry == mock_tool_registry


# Simple integration test that doesn't require external dependencies
def test_bigtool_module_imports():
    """Test that bigtool module can be imported successfully."""
    try:
        from app.core import bigtool_setup
        assert hasattr(bigtool_setup, 'BigToolManager')
        assert hasattr(bigtool_setup, 'create_bigtool_manager')
    except ImportError:
        pytest.skip("BigTool module dependencies not available")
