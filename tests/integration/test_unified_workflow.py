import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.agents.state import MathAgentState, WorkflowStatus, WorkflowSteps
from app.agents.graph import MathematicalAgentGraph
from app.agents.state_utils import create_initial_state
from app.core.exceptions import AgentError, ValidationError

# Import our professional mock infrastructure
from tests.fixtures.mock_factory import MockFactory, TestValidationHelpers


class TestUnifiedMathematicalWorkflow:
    """Test suite for the unified mathematical workflow with ZERO API calls."""
    
    @pytest.fixture
    def settings(self):
        """Create MOCK settings for testing - never uses real API keys."""
        return MockFactory.create_mock_settings()
    
    @pytest.fixture
    def tool_registry(self):
        """Create MOCK tool registry for testing."""
        return MockFactory.create_mock_tool_registry()
    
    @pytest.fixture
    def workflow_graph(self, settings, tool_registry):
        """Create MathematicalAgentGraph instance with MOCK dependencies."""
        return MathematicalAgentGraph(
            settings=settings,
            tool_registry=tool_registry
        )
    
    @pytest.fixture
    def sample_integral_problem(self):
        """Sample integral problem for testing."""
        return "Calculate the integral of x^2 from 0 to 2"
    
    @pytest.fixture
    def sample_initial_state(self, sample_integral_problem):
        """Create initial state for testing."""
        return create_initial_state(problem=sample_integral_problem)

    def test_workflow_graph_initialization(self, settings, tool_registry):
        """Test MathematicalAgentGraph initialization."""
        graph = MathematicalAgentGraph(
            settings=settings,
            tool_registry=tool_registry
        )
        
        assert graph.settings is not None
        assert graph.tool_registry is not None
        assert graph._workflow is None  # Not built yet
        assert graph._compiled_graph is None  # Not compiled yet

    def test_workflow_building(self, workflow_graph):
        """Test workflow graph building."""
        workflow = workflow_graph.build_workflow()
        
        assert workflow is not None
        assert workflow_graph._workflow is not None
        
        # Verify nodes are added
        node_names = list(workflow.nodes.keys())
        expected_nodes = [
            "analyze_problem", 
            "reasoning", 
            "execute_tools", 
            "validation", 
            "finalization"
        ]
        
        for node in expected_nodes:
            assert node in node_names

    def test_workflow_compilation(self, workflow_graph):
        """Test workflow graph compilation."""
        compiled_graph = workflow_graph.compile_graph()
        
        assert compiled_graph is not None
        assert workflow_graph._compiled_graph is not None

    @pytest.mark.asyncio
    async def test_simple_problem_solving_flow(self, workflow_graph, sample_initial_state):
        """Test complete problem solving flow with ZERO API calls."""
        
        # Use our professional mock infrastructure - NO real API calls
        with MockFactory.mock_all_api_calls() as mocks:
            
            # Build and compile workflow with mocked dependencies
            compiled_graph = workflow_graph.compile_graph()
            
            # Execute workflow - all API calls are mocked
            result = await compiled_graph.ainvoke(sample_initial_state)
            
            # Validate results came from mocks
            assert result is not None
            assert result.get('workflow_status') == WorkflowStatus.COMPLETED  # Fix: Use correct field name
            assert result.get('current_step') == WorkflowSteps.COMPLETE  # Professional pattern: Verify completion
            assert result.get('final_answer') is not None  # Professional pattern: Ensure answer exists
            assert result.get('confidence_score', 0) > 0.8
            
            # Validate NO real API calls were made
            TestValidationHelpers.assert_no_real_api_calls(mocks['llm_class'])
            TestValidationHelpers.assert_valid_mock_response(result)

    def test_workflow_error_handling(self, workflow_graph):
        """Test workflow error handling."""
        
        # Test with invalid state
        with pytest.raises(AgentError):
            workflow_graph.build_workflow = Mock(side_effect=Exception("Test error"))
            workflow_graph.compile_graph()

    @pytest.mark.asyncio
    async def test_workflow_with_tool_execution(self, workflow_graph, sample_initial_state):
        """Test workflow with tool execution path."""
        
        with patch('app.agents.nodes._get_chain_factory') as mock_get_factory:
            # Use professional mock factory pattern 
            mock_factory = MockFactory.create_mock_chain_factory()
            mock_get_factory.return_value = mock_factory
            
            # Mock BigTool with recommended tools
            with patch('app.agents.nodes.create_bigtool_manager') as mock_bigtool:
                mock_manager = Mock()
                mock_bigtool.return_value = mock_manager
                
                # Mock tool recommendation
                mock_tool = Mock()
                mock_tool.name = "integral_tool"
                mock_tool.similarity_score = 0.9
                mock_manager.search_tools.return_value = [mock_tool]
                
                # Mock tool registry
                with patch('app.agents.nodes.ToolRegistry') as mock_registry_class:
                    mock_registry = Mock()
                    mock_registry_class.return_value = mock_registry
                    
                    mock_tool_instance = AsyncMock()
                    mock_tool_instance.arun.return_value = "Tool result: 8/3"
                    mock_registry.get_tool.return_value = mock_tool_instance
                    
                    # Execute workflow
                    compiled_graph = workflow_graph.compile_graph()
                    result = await compiled_graph.ainvoke(sample_initial_state)
                    
                    # Validate professional completion pattern
                    assert result is not None
                    assert result.get('workflow_status') == WorkflowStatus.COMPLETED
                    assert result.get('current_step') == WorkflowSteps.COMPLETE
                    assert result.get('final_answer') is not None  # Should have final answer    def test_workflow_state_transitions(self, workflow_graph):
        """Test proper state transitions in workflow."""
        workflow = workflow_graph.build_workflow()
        
        # Verify entry point
        assert "analyze_problem" in workflow.nodes
        
        # Verify conditional edges structure (basic structure check)
        # Note: Detailed edge testing would require more complex mocking
        edges = workflow.edges
        assert len(edges) > 0  # Has conditional edges

    @pytest.mark.asyncio
    async def test_error_recovery_node(self, workflow_graph):
        """Test error recovery functionality."""
        
        # Create error state with proper loop detection fields
        error_state = {
            "current_problem": "test problem",
            "error": "Test error",
            "error_type": "test_error",
            "retry_count": 0,
            "iteration_count": 0,  # Professional pattern: Include iteration tracking
            "max_iterations": 10,  # Professional pattern: Include max iterations
            "current_step": WorkflowSteps.ERROR_RECOVERY
        }        # Test error recovery node directly
        from app.agents.nodes import error_recovery_node
        
        with patch('app.agents.nodes._get_chain_factory') as mock_get_factory:
            # Professional pattern: Use proper AsyncMock like in unit tests
            mock_factory = Mock()
            mock_recovery_chain = AsyncMock()
            mock_recovery_chain.ainvoke.return_value = {
                "action": "retry_reasoning",
                "note": "Retrying with simplified approach"
            }
            mock_factory.create_error_recovery_chain.return_value = mock_recovery_chain
            mock_get_factory.return_value = mock_factory

            result = await error_recovery_node(error_state)

            assert result['retry_count'] == 1
            assert result['iteration_count'] == 0  # Should preserve iteration count
            assert result['recovery_action'] == "retry_reasoning"
            assert result['current_step'] == WorkflowSteps.REASONING

    def test_professional_architecture_principles(self, workflow_graph):
        """Test that the architecture follows professional principles."""
        
        # Test DRY: Single workflow instance
        workflow1 = workflow_graph.build_workflow()
        workflow2 = workflow_graph.build_workflow()
        assert workflow1 is workflow2  # Same cached instance
        
        # Test KISS: Simple initialization
        simple_graph = MathematicalAgentGraph()
        assert simple_graph is not None
        
        # Test YAGNI: No unnecessary dependencies in constructor
        import inspect
        sig = inspect.signature(MathematicalAgentGraph.__init__)
        # All parameters should be optional (have defaults or None)
        for param in sig.parameters.values():
            if param.name != 'self':
                # Parameter should either have a default value (including None) or be empty
                has_default = param.default is not inspect.Parameter.empty
                assert has_default, f"Parameter {param.name} should have a default value"

    async def test_workflow_performance(self, workflow_graph, sample_initial_state):
        """Test workflow execution performance."""
        
        start_time = datetime.now()
        
        with patch('app.agents.chains.create_chain_factory') as mock_chain_factory:
            # Quick mock setup for performance test
            mock_factory = Mock()
            mock_chain_factory.return_value = mock_factory
            
            for chain_type in ['analysis', 'reasoning', 'validation', 'response']:
                mock_chain = AsyncMock()
                mock_chain.ainvoke.return_value = {"confidence": 0.8}
                getattr(mock_factory, f'create_{chain_type}_chain').return_value = mock_chain
            
            with patch('app.agents.nodes.create_bigtool_manager'):
                compiled_graph = workflow_graph.compile_graph()
                await compiled_graph.ainvoke(sample_initial_state)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete within reasonable time (mocked, so should be very fast)
        assert execution_time < 5.0  # 5 seconds max for mocked execution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
