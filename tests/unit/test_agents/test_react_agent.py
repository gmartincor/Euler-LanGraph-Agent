"""Unit tests for ReAct Mathematical Agent.

This module tests the core functionality of the ReAct agent following
professional testing patterns and ensuring integration with existing infrastructure.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
import asyncio
from datetime import datetime
from uuid import uuid4

# Local imports with error handling for missing dependencies
try:
    from app.agents.react_agent import ReactMathematicalAgent, create_react_agent
    from app.agents.chains import ChainFactory, create_chain_factory
    from app.agents.prompts import get_prompt_template, build_tool_description
    from app.core.config import Settings
    from app.tools.registry import ToolRegistry
    from app.models.agent_state import AgentMemory
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="LangGraph dependencies not available")
class TestReactMathematicalAgent:
    """Test cases for ReactMathematicalAgent."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.gemini_config = {
            "model_name": "gemini-1.5-pro",
            "api_key": "test-key",
            "temperature": 0.1,
            "max_tokens": 8192
        }
        settings.agent_max_iterations = 10
        settings.tool_search_top_k = 3
        return settings
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry for testing."""
        registry = Mock(spec=ToolRegistry)
        registry.list_tools.return_value = ["integral_tool", "plot_tool", "analysis_tool"]
        registry.search_tools.return_value = [
            {"tool_name": "integral_tool", "score": 0.9},
            {"tool_name": "plot_tool", "score": 0.7}
        ]
        
        # Mock individual tools
        mock_tool = Mock()
        mock_tool.name = "integral_tool"
        mock_tool.description = "Calculate integrals"
        mock_tool.usage_stats = {
            "usage_count": 10,
            "success_rate": 0.9,
            "average_execution_time": 1.5
        }
        mock_tool.execute.return_value = Mock(
            success=True,
            result="∫x²dx = x³/3 + C",
            error=None
        )
        
        registry.get_tool.return_value = mock_tool
        return registry
    
    @pytest.fixture
    def mock_checkpointer(self):
        """Create mock checkpointer for testing."""
        checkpointer = Mock()
        return checkpointer
    
    def test_agent_initialization(self, mock_settings, mock_tool_registry):
        """Test basic agent initialization."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry,
            session_id="test-session"
        )
        
        assert agent.session_id == "test-session"
        assert agent.settings == mock_settings
        assert agent.tool_registry == mock_tool_registry
        assert not agent.is_initialized()
    
    def test_agent_session_info(self, mock_settings, mock_tool_registry):
        """Test session info retrieval."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry,
            session_id="test-session"
        )
        
        info = agent.session_info
        assert info["session_id"] == "test-session"
        assert info["initialized"] is False
        assert "available_tools" in info
        assert "bigtool_enabled" in info
    
    def test_available_tools_property(self, mock_settings, mock_tool_registry):
        """Test available tools property."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        tools = agent.available_tools
        assert tools == ["integral_tool", "plot_tool", "analysis_tool"]
        mock_tool_registry.list_tools.assert_called_once()
    
    def test_extract_problem_from_messages(self, mock_settings, mock_tool_registry):
        """Test problem extraction from messages."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        # Mock messages
        mock_message = Mock()
        mock_message.content = "Calculate ∫x²dx"
        messages = [mock_message]
        
        problem = agent._extract_problem_from_messages(messages)
        assert problem == "Calculate ∫x²dx"
    
    def test_extract_tool_parameters(self, mock_settings, mock_tool_registry):
        """Test tool parameter extraction."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        # Mock state
        state = {
            "current_problem": "Calculate ∫x²dx",
            "mathematical_context": {"variable": "x"}
        }
        
        params = agent._extract_tool_parameters(state, "integral_tool")
        assert isinstance(params, dict)
        assert "expression" in params
    
    def test_extract_final_answer(self, mock_settings, mock_tool_registry):
        """Test final answer extraction."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        # Mock state with tool results
        state = {
            "tool_results": [
                {
                    "tool_name": "integral_tool",
                    "result": "x³/3 + C",
                    "success": True
                }
            ]
        }
        
        answer = agent._extract_final_answer(state)
        assert answer == "x³/3 + C"
    
    def test_conditional_edge_functions(self, mock_settings, mock_tool_registry):
        """Test conditional edge functions."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        # Test _should_use_tools
        state_with_error = {"error_count": 1}
        assert agent._should_use_tools(state_with_error) == "error"
        
        state_with_tool_mention = {"current_reasoning": "I need to use a tool"}
        assert agent._should_use_tools(state_with_tool_mention) == "use_tools"
        
        state_normal = {"current_reasoning": "This is normal reasoning"}
        assert agent._should_use_tools(state_normal) == "validate"
        
        # Test _should_continue_reasoning
        state_max_iterations = {"iteration_count": 15, "max_iterations": 10}
        assert agent._should_continue_reasoning(state_max_iterations) == "validate"
        
        # Test _should_finalize
        state_high_confidence = {"confidence_score": 0.9}
        assert agent._should_finalize(state_high_confidence) == "finalize"
        
        state_low_confidence = {"confidence_score": 0.5}
        assert agent._should_finalize(state_low_confidence) == "continue"
    
    @pytest.mark.asyncio
    async def test_solve_problem_not_initialized(self, mock_settings, mock_tool_registry):
        """Test solve_problem when agent is not initialized."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        with pytest.raises(Exception) as exc_info:
            await agent.solve_problem("Calculate ∫x²dx")
        
        assert "not initialized" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_conversation_history_no_checkpointer(self, mock_settings, mock_tool_registry):
        """Test conversation history without checkpointer."""
        agent = ReactMathematicalAgent(
            settings=mock_settings,
            tool_registry=mock_tool_registry
        )
        
        history = await agent.get_conversation_history()
        assert history == []


@pytest.mark.skipif(not DEPENDENCIES_AVAILABLE, reason="LangGraph dependencies not available")
class TestChainFactory:
    """Test cases for ChainFactory."""
    
    @pytest.fixture
    def mock_settings(self):
        """Create mock settings for testing."""
        settings = Mock()
        settings.gemini_config = {
            "model_name": "gemini-1.5-pro",
            "api_key": "test-key",
            "temperature": 0.1,
            "max_tokens": 8192
        }
        settings.tool_search_top_k = 3
        return settings
    
    @pytest.fixture
    def mock_tool_registry(self):
        """Create mock tool registry for testing."""
        registry = Mock(spec=ToolRegistry)
        registry.list_tools.return_value = ["integral_tool", "plot_tool"]
        registry.search_tools.return_value = [
            {"tool_name": "integral_tool", "score": 0.9}
        ]
        
        mock_tool = Mock()
        mock_tool.name = "integral_tool"
        mock_tool.description = "Calculate integrals"
        mock_tool.usage_stats = {"usage_count": 10, "success_rate": 0.9}
        registry.get_tool.return_value = mock_tool
        
        return registry
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM for testing."""
        llm = Mock()
        return llm
    
    def test_chain_factory_initialization(self, mock_settings, mock_tool_registry, mock_llm):
        """Test ChainFactory initialization."""
        factory = ChainFactory(
            settings=mock_settings,
            tool_registry=mock_tool_registry,
            llm=mock_llm
        )
        
        assert factory.settings == mock_settings
        assert factory.tool_registry == mock_tool_registry
        assert factory.llm == mock_llm
    
    def test_create_chain_factory_function(self, mock_settings, mock_tool_registry):
        """Test create_chain_factory function."""
        with patch('app.agents.chains.ChainFactory') as mock_factory_class:
            mock_factory_instance = Mock()
            mock_factory_class.return_value = mock_factory_instance
            
            result = create_chain_factory(mock_settings, mock_tool_registry)
            
            mock_factory_class.assert_called_once_with(mock_settings, mock_tool_registry, None)
            assert result == mock_factory_instance


class TestPrompts:
    """Test cases for prompt templates and utilities."""
    
    def test_get_prompt_template_valid(self):
        """Test getting valid prompt template."""
        template = get_prompt_template("mathematical_reasoning")
        assert isinstance(template, str)
        assert "mathematical problem" in template.lower()
    
    def test_get_prompt_template_invalid(self):
        """Test getting invalid prompt template."""
        with pytest.raises(KeyError) as exc_info:
            get_prompt_template("nonexistent_template")
        
        assert "not found" in str(exc_info.value)
    
    def test_build_tool_description(self):
        """Test building tool descriptions."""
        tools_info = {
            "integral_tool": {
                "description": "Calculate integrals",
                "capabilities": ["symbolic", "numerical"],
                "usage_stats": {"success_rate": 0.9}
            },
            "plot_tool": {
                "description": "Create plots"
            }
        }
        
        description = build_tool_description(tools_info)
        assert "integral_tool" in description
        assert "Calculate integrals" in description
        assert "plot_tool" in description
        assert "Success Rate: 90.0%" in description
    
    def test_build_tool_description_empty(self):
        """Test building tool descriptions with empty input."""
        description = build_tool_description({})
        assert description == ""


class TestAgentIntegration:
    """Integration tests for agent components."""
    
    def test_agent_creation_with_defaults(self):
        """Test agent creation with default parameters.""" 
        with patch('app.agents.react_agent.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_get_settings.return_value = mock_settings
            
            with patch('app.agents.react_agent.ReactMathematicalAgent._get_default_tool_registry') as mock_get_registry:
                mock_registry = Mock()
                mock_get_registry.return_value = mock_registry
                
                agent = ReactMathematicalAgent()
                
                assert agent.settings == mock_settings
                assert agent.tool_registry == mock_registry
                assert agent.session_id is not None
    
    def test_agent_error_handling(self):
        """Test agent error handling during initialization."""
        with patch('app.agents.react_agent.get_settings', side_effect=Exception("Config error")):
            # Should not raise exception during construction
            agent = ReactMathematicalAgent()
            # But should handle the error gracefully
            assert agent is not None


# === Performance and Edge Case Tests ===

class TestAgentPerformance:
    """Performance and edge case tests."""
    
    def test_large_message_history(self):
        """Test handling of large message history."""
        agent = ReactMathematicalAgent()
        
        # Create large number of mock messages
        messages = []
        for i in range(1000):
            mock_msg = Mock()
            mock_msg.content = f"Message {i}"
            messages.append(mock_msg)
        
        # Should handle gracefully
        problem = agent._extract_problem_from_messages(messages)
        assert problem == "Message 999"  # Should get the last message
    
    def test_empty_state_handling(self):
        """Test handling of empty or minimal state."""
        agent = ReactMathematicalAgent()
        
        # Test with empty state
        empty_state = {}
        
        # Should not crash
        result = agent._should_use_tools(empty_state)
        assert result in ["error", "use_tools", "validate"]
        
        result = agent._should_continue_reasoning(empty_state)
        assert result in ["error", "continue", "validate"]
    
    def test_malformed_tool_results(self):
        """Test handling of malformed tool results."""
        agent = ReactMathematicalAgent()
        
        # Test with malformed tool results
        state_malformed = {
            "tool_results": [
                {"malformed": "data"},
                None,
                {"tool_name": "test", "success": True}  # Valid one
            ]
        }
        
        # Should handle gracefully and extract what it can
        answer = agent._extract_final_answer(state_malformed)
        assert isinstance(answer, str)


# === Mock Classes for Testing ===

class MockAgent:
    """Mock agent for testing external integrations."""
    
    def __init__(self):
        self.initialized = False
        self.session_id = str(uuid4())
    
    async def initialize(self):
        self.initialized = True
    
    async def solve_problem(self, problem: str) -> Dict[str, Any]:
        return {
            "success": True,
            "final_answer": f"Mock solution for: {problem}",
            "confidence_score": 0.8
        }


# === Fixtures for Integration Tests ===

@pytest.fixture
def integration_agent():
    """Create agent for integration testing."""
    return MockAgent()


@pytest.mark.asyncio
async def test_agent_lifecycle(integration_agent):
    """Test complete agent lifecycle."""
    # Initialize
    await integration_agent.initialize()
    assert integration_agent.initialized
    
    # Solve problem
    result = await integration_agent.solve_problem("Calculate ∫x²dx")
    assert result["success"] is True
    assert "Mock solution" in result["final_answer"]
    assert result["confidence_score"] > 0
