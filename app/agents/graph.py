from typing import Any, Dict, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END

from ..core.logging import get_logger, log_function_call
from ..core.config import get_settings
from ..core.exceptions import AgentError
from ..tools.registry import ToolRegistry
from .state import MathAgentState, WorkflowSteps, WorkflowStatus

from .nodes import (
    analyze_problem_node,
    reasoning_node,
    semantic_filter_node,  # NEW: Semantic filtering node
    tool_execution_node,
    validation_node,
    finalization_node,
    error_recovery_node
)
from .conditions import (
    should_continue_reasoning,
    should_execute_tools,
    should_validate_result,
    should_finalize
)
from .checkpointer import PostgreSQLCheckpointer

logger = get_logger(__name__)


class MathematicalAgentGraph:
    """
    Professional LangGraph Mathematical Workflow Implementation.
    
    This class provides the complete mathematical problem-solving workflow
    using LangGraph's StateGraph pattern. 
    """
    
    def __init__(
        self,
        settings: Optional[Any] = None,
        tool_registry: Optional[ToolRegistry] = None,
        checkpointer: Optional[Any] = None
    ):
        """
        Initialize the mathematical workflow graph.
        
        Args:
            settings: Application settings (optional, will use default)
            tool_registry: Tool registry instance (optional, will create default)
            checkpointer: Graph state checkpointer (optional, will use memory)
        """
        self.settings = settings or get_settings()
        self.tool_registry = tool_registry or ToolRegistry()
        self.checkpointer = checkpointer
        
        # Initialize workflow components
        self._workflow = None
        self._compiled_graph = None
        
        logger.info("Mathematical agent graph initialized successfully")
    
    @log_function_call(logger)
    def build_workflow(self) -> StateGraph:
        """
        Build the complete mathematical problem-solving workflow.
        
        Creates a LangGraph StateGraph with all necessary nodes and edges
        for mathematical problem solving, following the unified architecture.
        
        Returns:
            StateGraph: The complete workflow graph
            
        Raises:
            AgentError: If workflow building fails
        """
        try:
            # Return cached workflow if available
            if self._workflow is not None:
                logger.debug("Returning cached workflow graph")
                return self._workflow
                
            logger.info("Building mathematical workflow graph...")
            
            # Create state graph
            workflow = StateGraph(MathAgentState)
            
            # Add workflow nodes 
            workflow.add_node("analyze_problem", analyze_problem_node)
            workflow.add_node("reasoning", reasoning_node)
            workflow.add_node("semantic_filter", semantic_filter_node)  # NEW NODE
            workflow.add_node("execute_tools", tool_execution_node)
            workflow.add_node("validation", validation_node)
            workflow.add_node("finalization", finalization_node)
            workflow.add_node("error_recovery", error_recovery_node)
            
            # Set entry point
            workflow.set_entry_point("analyze_problem")
            
            # Add conditional edges (workflow routing logic)
            workflow.add_conditional_edges(
                "analyze_problem",
                should_continue_reasoning,
                {
                    "continue": "reasoning",
                    "error": "finalization"
                }
            )
            
            workflow.add_conditional_edges(
                "reasoning",
                should_execute_tools,
                {
                    "execute_tools": "semantic_filter",  # MODIFIED: Go to filtering first
                    "validate": "validation",
                    "error": "finalization"
                }
            )
            
            # NEW EDGE: From semantic filtering to execution
            workflow.add_edge("semantic_filter", "execute_tools")
            
            workflow.add_conditional_edges(
                "execute_tools",
                should_validate_result,
                {
                    "validate": "validation",
                    "retry": "reasoning",
                    "error": "finalization"
                }
            )
            
            workflow.add_conditional_edges(
                "validation",
                should_finalize,
                {
                    "finalize": "finalization",
                    "retry": "reasoning",
                    "error": "finalization"
                }
            )
            
            # Final node goes to END
            workflow.add_edge("finalization", END)
            
            self._workflow = workflow
            logger.info("Mathematical workflow graph built successfully")
            
            return workflow
            
        except Exception as e:
            error_msg = f"Failed to build workflow: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    @log_function_call(logger)
    def compile_graph(self) -> Any:
        """
        Compile the workflow graph for execution.
        
        Compiles the StateGraph with checkpointer for persistent execution.
        
        Returns:
            CompiledGraph: The compiled graph ready for execution
            
        Raises:
            AgentError: If compilation fails
        """
        try:
            if not self._workflow:
                self.build_workflow()
            
            logger.info("Compiling mathematical workflow graph...")
            
            # Compile with checkpointer for persistence
            compiled_graph = self._workflow.compile(
                checkpointer=self.checkpointer
            )
            
            self._compiled_graph = compiled_graph
            logger.info("Mathematical workflow graph compiled successfully")
            
            return compiled_graph
            
        except Exception as e:
            error_msg = f"Failed to compile workflow: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    @log_function_call(logger)
    async def execute_workflow(
        self,
        initial_state: MathAgentState,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete mathematical workflow.
        
        Args:
            initial_state: Initial state for the workflow
            thread_id: Optional thread ID for conversation persistence
            
        Returns:
            Dict[str, Any]: Final workflow execution result
            
        Raises:
            AgentError: If workflow execution fails
        """
        try:
            if not self._compiled_graph:
                self.compile_graph()
            
            logger.info(f"Executing mathematical workflow for problem: {initial_state.get('current_problem', 'Unknown')[:50]}...")
            
            # Configure execution
            config = {"configurable": {"thread_id": thread_id or "default"}}
            
            # Execute workflow
            result = await self._compiled_graph.ainvoke(initial_state, config=config)
            
            logger.info("Mathematical workflow executed successfully")
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute workflow: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status and health information.
        
        Returns:
            Dict[str, Any]: Workflow status information
        """
        return {
            "workflow_built": self._workflow is not None,
            "graph_compiled": self._compiled_graph is not None,
            "settings_configured": self.settings is not None,
            "tools_available": len(self.tool_registry.get_all_tools()) if self.tool_registry else 0,
            "timestamp": datetime.utcnow().isoformat()
        }


# === Factory Functions for Clean Interface ===

def create_mathematical_agent_graph(
    settings: Optional[Any] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None
) -> MathematicalAgentGraph:
    """
    Factory function to create a mathematical agent graph.
    
    Provides a clean interface for creating workflow graphs with optional
    dependency injection for testing and customization.
    
    Args:
        settings: Application settings (optional)
        tool_registry: Tool registry instance (optional)
        checkpointer: Graph state checkpointer (optional)
        
    Returns:
        MathematicalAgentGraph: Configured workflow graph
    """
    return MathematicalAgentGraph(
        settings=settings,
        tool_registry=tool_registry,
        checkpointer=checkpointer
    )


def create_agent_graph(
    settings: Optional[Any] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None
) -> MathematicalAgentGraph:
    """
    Alias for create_mathematical_agent_graph for backward compatibility.
    
    Args:
        settings: Application settings (optional)
        tool_registry: Tool registry instance (optional)
        checkpointer: Graph state checkpointer (optional)
        
    Returns:
        MathematicalAgentGraph: Configured workflow graph
    """
    return create_mathematical_agent_graph(
        settings=settings,
        tool_registry=tool_registry,
        checkpointer=checkpointer
    )


def create_compiled_workflow(
    settings: Optional[Any] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[Any] = None
) -> Any:
    """
    Factory function to create a compiled workflow ready for execution.
    
    Provides a convenient interface for getting a ready-to-use compiled
    workflow without manual building and compilation steps.
    
    Args:
        settings: Application settings (optional)
        tool_registry: Tool registry instance (optional)
        checkpointer: Graph state checkpointer (optional)
        
    Returns:
        CompiledGraph: Ready-to-execute compiled workflow
    """
    graph = create_mathematical_agent_graph(
        settings=settings,
        tool_registry=tool_registry,
        checkpointer=checkpointer
    )
    
    return graph.compile_graph()
