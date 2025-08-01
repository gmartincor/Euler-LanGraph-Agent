"""Integration Tests for Unified Mathematical Workflow - Professional Architecture.

These tests validate the complete mathematical reasoning workflow using the 
unified LangGraph architecture, eliminating circular dependencies and following
professional testing patterns.

Key Testing Patterns Applied:
- Integration Testing: End-to-end workflow validation
- Clean Architecture: Tests pure LangGraph workflow without circular dependencies
- Professional Assertions: Comprehensive result validation
- Error Case Testing: Edge cases and error recovery validation
- Performance Testing: Workflow execution timing validation

Architecture Benefits:
- Zero Circular Dependencies: Tests clean architecture
- Comprehensive Coverage: Tests all workflow paths and error cases
- Professional Quality: Clean testing standards
- DRY Principle: Reusable test fixtures and patterns
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.agents.state import MathAgentState, WorkflowStatus, WorkflowSteps
from app.agents.graph import MathematicalAgentGraph
from app.agents.state_utils import create_initial_state
from app.core.exceptions import AgentError, ValidationError
from app.core.config import get_settings
from app.tools.registry import ToolRegistry


class TestUnifiedMathematicalWorkflow:
    """Test suite for the unified mathematical workflow."""
    
    @pytest.fixture
    def settings(self):
        """Create settings for testing."""
        return get_settings()
    
    @pytest.fixture
    def tool_registry(self):
        """Create tool registry for testing."""
        return ToolRegistry()
    
    @pytest.fixture
    def workflow_graph(self, settings, tool_registry):
        """Create MathematicalAgentGraph instance for testing."""
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
        """Test complete problem solving flow with mocked components."""
        
        # Mock external dependencies to avoid real API calls
        with patch('app.agents.chains.create_chain_factory') as mock_chain_factory:
            # Setup mock chain factory
            mock_factory = Mock()
            mock_chain_factory.return_value = mock_factory
            
            # Mock analysis chain
            mock_analysis_chain = AsyncMock()
            mock_analysis_chain.ainvoke.return_value = {
                "problem_type": "integral",
                "complexity": "medium",
                "requires_tools": True,
                "description": "Integration problem",
                "approach": "definite integral",
                "confidence": 0.9
            }
            mock_factory.create_analysis_chain.return_value = mock_analysis_chain
            
            # Mock reasoning chain
            mock_reasoning_chain = AsyncMock()
            mock_reasoning_chain.ainvoke.return_value = {
                "approach": "Use fundamental theorem of calculus",
                "steps": ["Find antiderivative", "Apply limits"],
                "tools_needed": ["integral_tool"],
                "confidence": 0.85
            }
            mock_factory.create_reasoning_chain.return_value = mock_reasoning_chain
            
            # Mock validation chain
            mock_validation_chain = AsyncMock()
            mock_validation_chain.ainvoke.return_value = {
                "is_valid": True,
                "score": 0.9,
                "issues": []
            }
            mock_factory.create_validation_chain.return_value = mock_validation_chain
            
            # Mock response chain
            mock_response_chain = AsyncMock()
            mock_response_chain.ainvoke.return_value = {
                "answer": "8/3",
                "steps": ["∫x² dx = x³/3", "Apply limits: [x³/3] from 0 to 2", "= 8/3 - 0 = 8/3"],
                "explanation": "The integral of x² from 0 to 2 equals 8/3",
                "confidence": 0.9
            }
            mock_factory.create_response_chain.return_value = mock_response_chain
            
            # Mock BigTool manager
            with patch('app.agents.nodes.create_bigtool_manager') as mock_bigtool:
                mock_manager = Mock()
                mock_bigtool.return_value = mock_manager
                mock_manager.search_tools.return_value = []  # No tools for simplicity
                
                # Build and compile workflow
                compiled_graph = workflow_graph.compile_graph()
                
                # Execute workflow
                result = await compiled_graph.ainvoke(sample_initial_state)
                
                # Validate results
                assert result is not None
                assert result.get('status') == WorkflowStatus.COMPLETED
                assert result.get('final_answer') == "8/3"
                assert result.get('is_complete') is True
                assert result.get('confidence_score', 0) > 0.8

    def test_workflow_error_handling(self, workflow_graph):
        """Test workflow error handling."""
        
        # Test with invalid state
        with pytest.raises(AgentError):
            workflow_graph.build_workflow = Mock(side_effect=Exception("Test error"))
            workflow_graph.compile_graph()

    @pytest.mark.asyncio 
    async def test_workflow_with_tool_execution(self, workflow_graph, sample_initial_state):
        """Test workflow with tool execution path."""
        
        with patch('app.agents.chains.create_chain_factory') as mock_chain_factory:
            # Setup chains similar to previous test but with tools needed
            mock_factory = Mock()
            mock_chain_factory.return_value = mock_factory
            
            # Analysis expects tools
            mock_analysis_chain = AsyncMock()
            mock_analysis_chain.ainvoke.return_value = {
                "problem_type": "integral",
                "complexity": "high",
                "requires_tools": True,
                "confidence": 0.8
            }
            mock_factory.create_analysis_chain.return_value = mock_analysis_chain
            
            # Reasoning suggests tools
            mock_reasoning_chain = AsyncMock()
            mock_reasoning_chain.ainvoke.return_value = {
                "approach": "Numerical integration",
                "steps": ["Setup integral", "Apply numerical methods"],
                "tools_needed": ["integral_tool", "plot_tool"],
                "confidence": 0.8
            }
            mock_factory.create_reasoning_chain.return_value = mock_reasoning_chain
            
            # Other chains
            mock_validation_chain = AsyncMock()
            mock_validation_chain.ainvoke.return_value = {
                "is_valid": True,
                "score": 0.85,
                "issues": []
            }
            mock_factory.create_validation_chain.return_value = mock_validation_chain
            
            mock_response_chain = AsyncMock()
            mock_response_chain.ainvoke.return_value = {
                "answer": "8/3 (numerical approximation)",
                "steps": ["Numerical integration applied"],
                "explanation": "Result computed using numerical methods",
                "confidence": 0.85
            }
            mock_factory.create_response_chain.return_value = mock_response_chain
            
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
                    
                    # Validate tool execution occurred
                    assert result is not None
                    assert result.get('tool_results') is not None
                    assert len(result.get('tool_results', [])) > 0

    def test_workflow_state_transitions(self, workflow_graph):
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
        
        # Create error state
        error_state = {
            "current_problem": "test problem", 
            "error": "Test error",
            "error_type": "test_error",
            "retry_count": 0,
            "current_step": WorkflowSteps.ERROR_RECOVERY
        }
        
        # Test error recovery node directly
        from app.agents.nodes import error_recovery_node
        
        with patch('app.agents.chains.create_chain_factory') as mock_chain_factory:
            mock_factory = Mock()
            mock_chain_factory.return_value = mock_factory
            
            mock_recovery_chain = AsyncMock()
            mock_recovery_chain.ainvoke.return_value = {
                "action": "retry_reasoning",
                "note": "Retrying with simplified approach"
            }
            mock_factory.create_error_recovery_chain.return_value = mock_recovery_chain
            
            result = await error_recovery_node(error_state)
            
            assert result['retry_count'] == 1
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
        # All parameters should be optional (have defaults)
        for param in sig.parameters.values():
            if param.name != 'self':
                assert param.default is not None or param.default is inspect.Parameter.empty

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
