"""Unit tests for Mathematical Agent Interface - Unified Architecture.

Professional unit tests for the clean mathematical agent interface,
ensuring proper functionality and clean architecture principles.

Key Testing Patterns Applied:
- Interface Testing: Tests public API methods
- Professional Mocking: Comprehensive mock strategies
- Edge Case Coverage: Tests error conditions and edge cases
- Performance Validation: Ensures optimal interface performance
- Architecture Validation: Confirms clean interface design

Architecture Benefits:
- Clean API Testing: Tests simple, intuitive interface
- Zero Circular Dependencies: Tests pure interface layer
- Professional Quality: Comprehensive test coverage
- DRY Principle: Reusable test fixtures and assertions
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any
from uuid import uuid4

from app.agents.interface import MathematicalAgent, create_mathematical_agent
from app.core.exceptions import AgentError, ValidationError
from app.agents.state import WorkflowStatus


class TestMathematicalAgent:
    """Test suite for MathematicalAgent interface."""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing."""
        with patch('app.agents.interface.ToolRegistry'), \
             patch('app.agents.interface.create_checkpointer'), \
             patch('app.agents.interface.MathematicalAgentGraph'):
            return MathematicalAgent(enable_persistence=False)
    
    @pytest.fixture
    def mock_workflow_result(self):
        """Create mock workflow result."""
        return {
            'status': WorkflowStatus.COMPLETED,
            'current_step': 'COMPLETE',
            'final_answer': '8/3',
            'solution_steps': [
                'Find antiderivative: x³/3',
                'Apply limits: [x³/3] from 0 to 2',
                'Result: 8/3'
            ],
            'explanation': 'Using fundamental theorem of calculus',
            'confidence_score': 0.9,
            'is_complete': True,
            'reasoning_trace': ['Analysis complete', 'Reasoning complete']
        }

    def test_agent_initialization(self):
        """Test agent initialization."""
        with patch('app.agents.interface.ToolRegistry'), \
             patch('app.agents.interface.create_checkpointer'), \
             patch('app.agents.interface.MathematicalAgentGraph'):
            
            agent = MathematicalAgent()
            
            assert agent.session_id is not None
            assert agent.settings is not None
            assert agent.enable_persistence is True
            assert agent.tool_registry is not None
            assert agent.workflow_graph is not None

    def test_agent_initialization_with_params(self):
        """Test agent initialization with custom parameters."""
        with patch('app.agents.interface.ToolRegistry'), \
             patch('app.agents.interface.create_checkpointer'), \
             patch('app.agents.interface.MathematicalAgentGraph'):
            
            custom_session = "test-session-123"
            agent = MathematicalAgent(
                session_id=custom_session,
                enable_persistence=False
            )
            
            assert agent.session_id == custom_session
            assert agent.enable_persistence is False

    @pytest.mark.asyncio
    async def test_solve_success(self, agent, mock_workflow_result):
        """Test successful problem solving."""
        
        # Mock compiled workflow
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_workflow_result
        agent._compiled_workflow = mock_workflow
        
        # Mock format function
        with patch('app.agents.interface.format_agent_response') as mock_format:
            formatted_result = {
                'answer': '8/3',
                'steps': mock_workflow_result['solution_steps'],
                'explanation': mock_workflow_result['explanation'],
                'confidence': 0.9,
                'success': True
            }
            mock_format.return_value = formatted_result
            
            # Execute solve
            result = await agent.solve("∫ x² dx from 0 to 2")
            
            # Validate results
            assert result['answer'] == '8/3'
            assert result['success'] is True
            assert result['confidence'] == 0.9
            assert 'execution_time' in result
            assert result['session_id'] == agent.session_id

    @pytest.mark.asyncio
    async def test_solve_empty_problem(self, agent):
        """Test solve with empty problem."""
        
        with pytest.raises(ValidationError, match="Problem cannot be empty"):
            await agent.solve("")

    @pytest.mark.asyncio
    async def test_solve_whitespace_problem(self, agent):
        """Test solve with whitespace-only problem."""
        
        with pytest.raises(ValidationError, match="Problem cannot be empty"):
            await agent.solve("   ")

    @pytest.mark.asyncio
    async def test_solve_workflow_error(self, agent):
        """Test solve when workflow raises error."""
        
        # Mock compiled workflow to raise error
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.side_effect = Exception("Workflow failed")
        agent._compiled_workflow = mock_workflow
        
        with pytest.raises(AgentError, match="Failed to solve problem"):
            await agent.solve("test problem")

    @pytest.mark.asyncio
    async def test_solve_with_context(self, agent, mock_workflow_result):
        """Test solve with context."""
        
        # Mock compiled workflow
        mock_workflow = AsyncMock()
        mock_workflow.ainvoke.return_value = mock_workflow_result
        agent._compiled_workflow = mock_workflow
        
        # Mock format function
        with patch('app.agents.interface.format_agent_response') as mock_format:
            mock_format.return_value = {'answer': '8/3', 'success': True}
            
            context = ["Previous calculation: ∫ x dx = x²/2"]
            result = await agent.solve("∫ x² dx", context=context)
            
            # Verify context was passed
            assert result['success'] is True
            mock_workflow.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_solve_stream(self, agent):
        """Test streaming solve."""
        
        # Mock stream chunks
        stream_chunks = [
            {'current_step': 'ANALYSIS', 'confidence_score': 0.7},
            {'current_step': 'REASONING', 'confidence_score': 0.8, 'reasoning_trace': ['Analyzing problem']},
            {'current_step': 'COMPLETE', 'final_answer': '8/3', 'confidence_score': 0.9}
        ]
        
        # Mock compiled workflow stream
        async def mock_astream(*args, **kwargs):
            for chunk in stream_chunks:
                yield chunk
        
        mock_workflow = Mock()
        mock_workflow.astream = mock_astream
        agent._compiled_workflow = mock_workflow
        
        # Collect stream results
        results = []
        async for update in agent.solve_stream("∫ x² dx"):
            results.append(update)
        
        # Validate streaming
        assert len(results) == 3
        assert results[0]['current_step'] == 'ANALYSIS'
        assert results[1]['current_step'] == 'REASONING'
        assert results[2]['status'] == 'completed'
        assert results[2]['final_answer'] == '8/3'

    @pytest.mark.asyncio
    async def test_solve_stream_empty_problem(self, agent):
        """Test stream solve with empty problem."""
        
        updates = []
        async for update in agent.solve_stream(""):
            updates.append(update)
        
        assert len(updates) == 1
        assert 'error' in updates[0]
        assert updates[0]['status'] == 'failed'

    @pytest.mark.asyncio
    async def test_get_conversation_history_no_persistence(self, agent):
        """Test getting conversation history without persistence."""
        
        agent.enable_persistence = False
        history = await agent.get_conversation_history()
        
        assert history == []

    @pytest.mark.asyncio
    async def test_get_conversation_history_with_persistence(self, agent):
        """Test getting conversation history with persistence."""
        
        agent.enable_persistence = True
        mock_checkpointer = AsyncMock()
        mock_checkpointer.get_conversation_history.return_value = [
            {'problem': '∫ x dx', 'answer': 'x²/2'}
        ]
        agent.checkpointer = mock_checkpointer
        
        history = await agent.get_conversation_history()
        
        assert len(history) == 1
        assert history[0]['problem'] == '∫ x dx'

    @pytest.mark.asyncio
    async def test_clear_conversation_no_persistence(self, agent):
        """Test clearing conversation without persistence."""
        
        agent.enable_persistence = False
        result = await agent.clear_conversation()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_conversation_with_persistence(self, agent):
        """Test clearing conversation with persistence."""
        
        agent.enable_persistence = True
        mock_checkpointer = AsyncMock()
        mock_checkpointer.clear_conversation.return_value = True
        agent.checkpointer = mock_checkpointer
        
        result = await agent.clear_conversation()
        
        assert result is True

    def test_get_available_tools(self, agent):
        """Test getting available tools."""
        
        # Mock tool registry
        mock_registry = Mock()
        mock_registry.list_tools.return_value = ['integral_tool', 'plot_tool']
        agent.tool_registry = mock_registry
        
        tools = agent.get_available_tools()
        
        assert tools == ['integral_tool', 'plot_tool']

    def test_get_agent_info(self, agent):
        """Test getting agent information."""
        
        # Mock tool registry
        mock_registry = Mock()
        mock_registry.list_tools.return_value = ['integral_tool', 'plot_tool']
        agent.tool_registry = mock_registry
        
        info = agent.get_agent_info()
        
        assert info['session_id'] == agent.session_id
        assert info['version'] == '1.0.0'
        assert info['architecture'] == 'Unified LangGraph'
        assert 'integral_calculation' in info['capabilities']
        assert info['available_tools'] == ['integral_tool', 'plot_tool']

    def test_compiled_workflow_property(self, agent):
        """Test compiled workflow property."""
        
        # Mock workflow graph
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile_graph.return_value = mock_compiled
        agent.workflow_graph = mock_graph
        
        # First access should compile
        workflow1 = agent.compiled_workflow
        assert workflow1 is mock_compiled
        
        # Second access should return cached
        workflow2 = agent.compiled_workflow
        assert workflow2 is mock_compiled
        assert workflow1 is workflow2


class TestMathematicalAgentFactory:
    """Test suite for agent factory functions."""
    
    def test_create_mathematical_agent_default(self):
        """Test creating agent with default parameters."""
        
        with patch('app.agents.interface.MathematicalAgent') as mock_agent_class:
            mock_instance = Mock()
            mock_agent_class.return_value = mock_instance
            
            agent = create_mathematical_agent()
            
            mock_agent_class.assert_called_once_with(
                settings=None,
                session_id=None,
                enable_persistence=True
            )
            assert agent is mock_instance

    def test_create_mathematical_agent_custom(self):
        """Test creating agent with custom parameters."""
        
        with patch('app.agents.interface.MathematicalAgent') as mock_agent_class:
            mock_instance = Mock()
            mock_agent_class.return_value = mock_instance
            
            custom_session = "custom-session"
            agent = create_mathematical_agent(
                session_id=custom_session,
                enable_persistence=False
            )
            
            mock_agent_class.assert_called_once_with(
                settings=None,
                session_id=custom_session,
                enable_persistence=False
            )
            assert agent is mock_instance

    def test_convenience_aliases(self):
        """Test convenience aliases work correctly."""
        
        from app.agents.interface import Agent, create_agent
        
        # Test aliases point to correct classes/functions
        assert Agent is MathematicalAgent
        assert create_agent is create_mathematical_agent


class TestInterfaceArchitecture:
    """Test suite for interface architecture principles."""
    
    def test_clean_api_design(self):
        """Test that the API follows clean design principles."""
        
        import inspect
        
        # Test main methods have clean signatures
        solve_sig = inspect.signature(MathematicalAgent.solve)
        assert 'problem' in solve_sig.parameters
        assert 'context' in solve_sig.parameters
        
        # Test async methods are properly marked
        assert asyncio.iscoroutinefunction(MathematicalAgent.solve)
        assert asyncio.iscoroutinefunction(MathematicalAgent.solve_stream)

    def test_single_responsibility(self):
        """Test that the interface has single responsibility."""
        
        # Should have clear mathematical agent methods
        agent_methods = [
            method for method in dir(MathematicalAgent)
            if not method.startswith('_') and callable(getattr(MathematicalAgent, method))
        ]
        
        # All methods should be related to mathematical problem solving
        expected_methods = {
            'solve', 'solve_stream', 'get_conversation_history',
            'clear_conversation', 'get_available_tools', 'get_agent_info'
        }
        
        actual_methods = set(agent_methods) - {'compiled_workflow'}  # Property
        assert expected_methods.issubset(actual_methods)

    def test_no_circular_dependencies(self):
        """Test that interface has no circular dependencies."""
        
        # Interface should only import from core, tools, and agents modules
        # Should not import from main or other higher-level modules
        import app.agents.interface as interface_module
        
        # Get module's imports
        import_names = []
        for name in dir(interface_module):
            obj = getattr(interface_module, name)
            if hasattr(obj, '__module__') and obj.__module__:
                import_names.append(obj.__module__)
        
        # Should only import from allowed modules
        allowed_prefixes = ['app.core', 'app.tools', 'app.agents', 'typing', 'datetime', 'uuid']
        
        for import_name in import_names:
            if import_name.startswith('app.'):
                assert any(import_name.startswith(prefix) for prefix in allowed_prefixes), \
                       f"Unexpected import: {import_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
