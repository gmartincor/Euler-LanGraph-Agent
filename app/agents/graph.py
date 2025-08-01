"""LangGraph Workflow Orchestration for Mathematical ReAct Agent.

This module provides the professional StateGraph orchestration layer that extracts
and reuses existing workflow logic from ReactMathematicalAgent without duplication.

Key Design Patterns Applied:
- Facade Pattern: Simplified interface for complex workflow orchestration
- Factory Pattern: Centralized graph creation and configuration
- Dependency Injection: Agent dependencies injected cleanly
- Single Responsibility: Each class has one clear orchestration purpose
- Professional Error Handling: Comprehensive exception management
- Async/Await Optimization: Performance-optimized async operations

Architecture Benefits:
- Zero Code Duplication: Reuses existing ReactMathematicalAgent implementations
- Enhanced Modularity: Clean separation between logic and orchestration  
- Improved Testability: Standalone graph for individual testing
- Professional Quality: Maintains existing tested behavior
- Future-Proof Design: Easy to extend with additional workflow patterns
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ValidationError
from .state import MathAgentState
from .nodes import (
    analyze_problem_node,
    reasoning_node,
    tool_action_node,
    validation_node,
    final_response_node,
    error_recovery_node
)
from .conditions import (
    should_use_tools,
    should_continue_reasoning,
    should_finalize,
    should_retry
)

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from .react_agent import ReactMathematicalAgent

logger = get_logger(__name__)


class MathematicalAgentGraph:
    """
    Professional LangGraph orchestration for mathematical reasoning workflows.
    
    This class provides a clean, modular interface for creating and managing
    LangGraph StateGraph instances that orchestrate mathematical problem-solving
    workflows. It extracts and reuses existing ReactMathematicalAgent logic
    without duplication, following DRY and KISS principles.
    
    Key Responsibilities:
    - StateGraph creation and configuration
    - Node orchestration using extracted functions
    - Conditional edge configuration using extracted logic
    - Professional error handling and logging
    - Integration with checkpoint savers for persistence
    
    Design Philosophy:
    - EXTRACT, DON'T DUPLICATE: Reuses all existing tested implementations
    - SINGLE RESPONSIBILITY: Focus only on workflow orchestration
    - PROFESSIONAL QUALITY: Comprehensive error handling and logging
    - PERFORMANCE OPTIMIZED: Minimal overhead over direct agent usage
    """
    
    def __init__(self, agent: "ReactMathematicalAgent"):
        """
        Initialize the mathematical agent graph orchestrator.
        
        Args:
            agent: ReactMathematicalAgent instance with all workflow logic
        """
        if not agent:
            raise ValueError("Agent instance is required for graph orchestration")
            
        self.agent = agent
        self._graph_cache: Optional[StateGraph] = None
        
        logger.info(f"MathematicalAgentGraph initialized for session: {agent.session_id}")
    
    @log_function_call(logger)
    def build_graph(self) -> StateGraph:
        """
        Build StateGraph using extracted nodes and existing agent logic.
        
        This method creates a complete LangGraph StateGraph by orchestrating
        the extracted workflow nodes and conditional edge functions. It follows
        the ReAct pattern: Analyze -> Reason -> Act -> Validate -> Respond.
        
        Returns:
            StateGraph: Complete mathematical reasoning workflow
            
        Raises:
            AgentError: If graph creation fails
            ValidationError: If agent state is invalid
        """
        try:
            # Create the state graph with proper type checking
            workflow = StateGraph(MathAgentState)
            
            # === Add Workflow Nodes ===
            # Extract existing node implementations without duplication
            
            workflow.add_node(
                "analyze_problem", 
                lambda state: analyze_problem_node(state, self.agent)
            )
            
            workflow.add_node(
                "reasoning", 
                lambda state: reasoning_node(state, self.agent)
            )
            
            workflow.add_node(
                "tool_action", 
                lambda state: tool_action_node(state, self.agent)
            )
            
            workflow.add_node(
                "validation", 
                lambda state: validation_node(state, self.agent)
            )
            
            workflow.add_node(
                "final_response", 
                lambda state: final_response_node(state, self.agent)
            )
            
            workflow.add_node(
                "error_recovery", 
                lambda state: error_recovery_node(state, self.agent)
            )
            
            # === Configure Workflow Flow ===
            # Define the entry point
            workflow.set_entry_point("analyze_problem")
            
            # === Add Conditional Edges ===
            # Extract existing conditional logic without duplication
            
            # Fixed edge from problem analysis to reasoning
            workflow.add_edge("analyze_problem", "reasoning")
            
            # Conditional edges from reasoning
            workflow.add_conditional_edges(
                "reasoning",
                lambda state: should_use_tools(state, self.agent),
                {
                    "use_tools": "tool_action",
                    "validate": "validation", 
                    "error": "error_recovery"
                }
            )
            
            # Conditional edges from tool action
            workflow.add_conditional_edges(
                "tool_action",
                lambda state: should_continue_reasoning(state, self.agent),
                {
                    "continue": "reasoning",
                    "validate": "validation",
                    "error": "error_recovery"
                }
            )
            
            # Conditional edges from validation
            workflow.add_conditional_edges(
                "validation",
                lambda state: should_finalize(state, self.agent),
                {
                    "finalize": "final_response",
                    "continue": "reasoning",
                    "error": "error_recovery"
                }
            )
            
            # Conditional edges from error recovery
            workflow.add_conditional_edges(
                "error_recovery",
                lambda state: should_retry(state, self.agent),
                {
                    "retry": "reasoning",
                    "finalize": "final_response"
                }
            )
            
            # Fixed edge to end
            workflow.add_edge("final_response", END)
            
            # Cache the graph for reuse
            self._graph_cache = workflow
            
            logger.info("LangGraph StateGraph created successfully with ReAct pattern")
            return workflow
            
        except Exception as e:
            logger.error(f"Failed to build StateGraph: {e}")
            raise AgentError(f"Graph creation failed: {e}") from e
    
    @log_function_call(logger)
    def compile_graph(
        self, 
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None
    ) -> Any:
        """
        Compile StateGraph with optional checkpointing and interrupts.
        
        Args:
            checkpointer: Optional checkpoint saver for persistence
            interrupt_before: Optional list of nodes to interrupt before
            interrupt_after: Optional list of nodes to interrupt after
            
        Returns:
            CompiledGraph: Compiled and ready-to-use graph
            
        Raises:
            AgentError: If compilation fails
        """
        try:
            # Get or build the graph
            graph = self._graph_cache or self.build_graph()
            
            # Compile with options
            compiled = graph.compile(
                checkpointer=checkpointer,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after
            )
            
            logger.info(
                f"StateGraph compiled successfully "
                f"(checkpointer={'enabled' if checkpointer else 'disabled'})"
            )
            
            return compiled
            
        except Exception as e:
            logger.error(f"Failed to compile StateGraph: {e}")
            raise AgentError(f"Graph compilation failed: {e}") from e
    
    @log_function_call(logger)
    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current graph configuration.
        
        Returns:
            Dict containing graph metadata and configuration info
        """
        try:
            graph = self._graph_cache or self.build_graph()
            
            return {
                "session_id": self.agent.session_id,
                "nodes": [
                    "analyze_problem",
                    "reasoning", 
                    "tool_action",
                    "validation",
                    "final_response",
                    "error_recovery"
                ],
                "entry_point": "analyze_problem",
                "end_points": ["final_response"],
                "conditional_edges": {
                    "reasoning": ["use_tools", "validate", "error"],
                    "tool_action": ["continue", "validate", "error"],
                    "validation": ["finalize", "continue", "error"],
                    "error_recovery": ["retry", "finalize"]
                },
                "agent_tools": len(self.agent.available_tools) if hasattr(self.agent, 'available_tools') else 0,
                "created_at": datetime.utcnow().isoformat(),
                "graph_cached": self._graph_cache is not None
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph info: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear the cached graph to force rebuilding on next access."""
        self._graph_cache = None
        logger.debug("Graph cache cleared")


# === Factory Functions ===

@log_function_call(logger)
def create_mathematical_agent_graph(agent: "ReactMathematicalAgent") -> MathematicalAgentGraph:
    """
    Factory function to create MathematicalAgentGraph instances.
    
    Args:
        agent: ReactMathematicalAgent instance
        
    Returns:
        MathematicalAgentGraph: Configured graph orchestrator
        
    Raises:
        ValueError: If agent is invalid
        AgentError: If creation fails
    """
    try:
        if not agent:
            raise ValueError("Valid ReactMathematicalAgent instance required")
            
        graph_orchestrator = MathematicalAgentGraph(agent)
        
        logger.info(f"MathematicalAgentGraph created for session: {agent.session_id}")
        return graph_orchestrator
        
    except Exception as e:
        logger.error(f"Failed to create MathematicalAgentGraph: {e}")
        raise AgentError(f"Graph orchestrator creation failed: {e}") from e


@log_function_call(logger)
def create_compiled_workflow(
    agent: "ReactMathematicalAgent",
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None
) -> Any:
    """
    Convenience function to create and compile a mathematical workflow in one step.
    
    Args:
        agent: ReactMathematicalAgent instance
        checkpointer: Optional checkpoint saver for persistence
        interrupt_before: Optional list of nodes to interrupt before
        interrupt_after: Optional list of nodes to interrupt after
        
    Returns:
        CompiledGraph: Ready-to-use compiled workflow
        
    Raises:
        AgentError: If creation or compilation fails
    """
    try:
        graph_orchestrator = create_mathematical_agent_graph(agent)
        compiled_workflow = graph_orchestrator.compile_graph(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after
        )
        
        logger.info(
            f"Compiled workflow created for session: {agent.session_id} "
            f"(checkpointer={'enabled' if checkpointer else 'disabled'})"
        )
        
        return compiled_workflow
        
    except Exception as e:
        logger.error(f"Failed to create compiled workflow: {e}")
        raise AgentError(f"Compiled workflow creation failed: {e}") from e
