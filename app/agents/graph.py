"""Professional LangGraph StateGraph Implementation.

This module orchestrates the mathematical reasoning workflow using LangGraph's
StateGraph pattern. It integrates all core components without circular dependencies.

Key Architecture Principles:
- Immutable State: State transitions are pure and traceable
- Compositional Design: Uses core components via dependency injection
- Professional Error Handling: Comprehensive recovery mechanisms
- Performance Optimized: Efficient workflow orchestration
"""

from typing import Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

from ..core.logging import get_logger
from ..core.exceptions import AgentError, ValidationError
from .checkpointer import AgentCheckpointer
from .state import MathAgentState, WorkflowSteps, WorkflowStatus
from .nodes import (
    analyze_problem_node,
    reasoning_node,
    tool_selection_node,
    tool_execution_node,
    validation_node,
    error_recovery_node,
    finalization_node
)
from .workflow_conditions import (
    should_use_tools,
    should_continue_reasoning,
    should_finalize,
    should_retry,
    workflow_complete
)

logger = get_logger(__name__)


class MathematicalWorkflowGraph:
    """
    Professional LangGraph workflow orchestrator for mathematical reasoning.
    
    This class manages the complete mathematical reasoning workflow using
    LangGraph's StateGraph pattern with professional error handling and
    state management.
    
    Architecture Benefits:
    - No Circular Dependencies: Clean dependency injection
    - Professional Quality: Production-ready workflow management
    - Testable: Each component can be tested independently
    - Maintainable: Clear separation of concerns
    """
    
    def __init__(self, checkpointer: Optional[AgentCheckpointer] = None):
        """Initialize the mathematical workflow graph."""
        self.logger = logger
        self.checkpointer = checkpointer or MemorySaver()
        self._graph = None
        self._compiled_graph = None
        
    def create_graph(self) -> StateGraph:
        """
        Create and configure the LangGraph StateGraph.
        
        Returns:
            StateGraph: Configured mathematical reasoning workflow
        """
        try:
            # Create the StateGraph with our state schema
            graph = StateGraph(MathAgentState)
            
            # Add all workflow nodes
            self._add_workflow_nodes(graph)
            
            # Configure workflow edges and conditions
            self._configure_workflow_edges(graph)
            
            # Set entry point
            graph.set_entry_point("analyze_problem")
            
            self._graph = graph
            return graph
            
        except Exception as e:
            self.logger.error(f"Error creating workflow graph: {e}")
            raise AgentError(f"Failed to create workflow graph: {e}")
    
    def compile_graph(self) -> Any:
        """
        Compile the graph for execution.
        
        Returns:
            Compiled graph ready for execution
        """
        try:
            if not self._graph:
                self.create_graph()
            
            # Compile with checkpointer for persistence
            self._compiled_graph = self._graph.compile(
                checkpointer=self.checkpointer
            )
            
            self.logger.info("Mathematical reasoning workflow graph compiled successfully")
            return self._compiled_graph
            
        except Exception as e:
            self.logger.error(f"Error compiling workflow graph: {e}")
            raise AgentError(f"Failed to compile workflow graph: {e}")
    
    def _add_workflow_nodes(self, graph: StateGraph) -> None:
        """Add all workflow nodes to the graph."""
        try:
            # Core workflow nodes
            graph.add_node("analyze_problem", analyze_problem_node)
            graph.add_node("reasoning", reasoning_node)
            graph.add_node("tool_selection", tool_selection_node)
            graph.add_node("tool_execution", tool_execution_node)
            graph.add_node("validation", validation_node)
            graph.add_node("finalize", finalization_node)
            
            # Error handling nodes
            graph.add_node("error_recovery", error_recovery_node)
            
            self.logger.debug("All workflow nodes added successfully")
            
        except Exception as e:
            self.logger.error(f"Error adding workflow nodes: {e}")
            raise AgentError(f"Failed to add workflow nodes: {e}")
    
    def _configure_workflow_edges(self, graph: StateGraph) -> None:
        """Configure the workflow edges and conditional logic."""
        try:
            # Main workflow edges
            graph.add_edge("analyze_problem", "reasoning")
            
            # Conditional edges for reasoning loop
            graph.add_conditional_edges(
                "reasoning",
                should_continue_reasoning,
                {
                    "continue": "tool_selection",
                    "finalize": "validation",
                    "error": "error_recovery"
                }
            )
            
            # Tool selection to execution
            graph.add_edge("tool_selection", "tool_execution")
            
            # Conditional edges for tool usage
            graph.add_conditional_edges(
                "tool_execution",
                should_use_tools,
                {
                    "use_tools": "reasoning",  # Back to reasoning with tool results
                    "validate": "validation",
                    "select_tools": "tool_selection",
                    "error": "error_recovery"
                }
            )
            
            # Validation conditional edges
            graph.add_conditional_edges(
                "validation",
                should_finalize,
                {
                    "finalize": "finalize",
                    "continue": "reasoning",
                    "error": "error_recovery"
                }
            )
            
            # Error recovery conditional edges
            graph.add_conditional_edges(
                "error_recovery",
                should_retry,
                {
                    "retry": "reasoning",
                    "recover": "tool_selection",
                    "fail": "finalize"  # Graceful failure
                }
            )
            
            # Finalize leads to END
            graph.add_edge("finalize", END)
            
            self.logger.debug("All workflow edges configured successfully")
            
        except Exception as e:
            self.logger.error(f"Error configuring workflow edges: {e}")
            raise AgentError(f"Failed to configure workflow edges: {e}")
    
    async def execute_workflow(
        self,
        initial_state: MathAgentState,
        config: Optional[Dict[str, Any]] = None
    ) -> MathAgentState:
        """
        Execute the complete mathematical reasoning workflow.
        
        Args:
            initial_state: Initial state for the workflow
            config: Optional configuration for execution
            
        Returns:
            MathAgentState: Final state after workflow completion
        """
        try:
            if not self._compiled_graph:
                self.compile_graph()
            
            # Set default config
            execution_config = config or {
                "configurable": {"thread_id": "mathematical_reasoning_session"}
            }
            
            self.logger.info("Starting mathematical reasoning workflow execution")
            
            # Execute the workflow
            final_state = None
            step_count = 0
            max_steps = 50  # Safety limit
            
            async for state in self._compiled_graph.astream(
                initial_state, 
                config=execution_config
            ):
                step_count += 1
                final_state = state
                
                # Log progress
                current_step = state.get('current_step', 'unknown')
                self.logger.debug(f"Workflow step {step_count}: {current_step}")
                
                # Safety check
                if step_count >= max_steps:
                    self.logger.warning(f"Workflow exceeded maximum steps ({max_steps})")
                    break
                
                # Check for completion
                if workflow_complete(state):
                    self.logger.info("Workflow completed successfully")
                    break
            
            return final_state or initial_state
            
        except Exception as e:
            self.logger.error(f"Error executing workflow: {e}")
            raise AgentError(f"Workflow execution failed: {e}")
    
    def get_workflow_status(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Get detailed workflow status information.
        
        Args:
            state: Current workflow state
            
        Returns:
            Dict: Detailed status information
        """
        try:
            return {
                "current_step": state.get('current_step', 'unknown'),
                "workflow_status": state.get('workflow_status', WorkflowStatus.ACTIVE),
                "iteration_count": state.get('iteration_count', 0),
                "error_count": state.get('error_count', 0),
                "confidence_score": state.get('confidence_score', 0.0),
                "has_final_answer": state.get('final_answer') is not None,
                "selected_tools": len(state.get('selected_tools', [])),
                "tool_results": len(state.get('tool_results', {})),
                "reasoning_chain_length": len(state.get('reasoning_chain', []))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}


# Factory function for easy instantiation
def create_mathematical_workflow_graph(
    checkpointer: Optional[AgentCheckpointer] = None
) -> MathematicalWorkflowGraph:
    """
    Factory function to create a mathematical workflow graph.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        MathematicalWorkflowGraph: Configured workflow graph
    """
    return MathematicalWorkflowGraph(checkpointer=checkpointer)
