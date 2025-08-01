"""LangGraph Workflow Nodes for Mathematical ReAct Agent.

This module contains the extracted workflow node implementations from ReactMathematicalAgent,
following professional design patterns and DRY principles. Each node is a standalone function
that can be tested independently while reusing the existing tested implementations.

Key Design Patterns Applied:
- Strategy Pattern: Each node implements a specific strategy
- Dependency Injection: Agent dependencies injected as parameters
- Single Responsibility: Each node has one clear purpose
- Professional Error Handling: Comprehensive exception management
- Async/Await Optimization: Performance-optimized async operations

Architecture Benefits:
- Zero Code Duplication: Reuses existing ReactMathematicalAgent implementations
- Enhanced Testability: Standalone functions for individual testing
- Improved Modularity: Clear separation of workflow concerns
- Professional Quality: Maintains existing tested behavior
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ToolError, ValidationError
from .state import MathAgentState

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from .react_agent import ReactMathematicalAgent

logger = get_logger(__name__)


# === Core Workflow Nodes ===

@log_function_call(logger)
async def analyze_problem_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Mathematical problem analysis workflow node.
    
    This node extracts and reuses the existing problem analysis logic
    from ReactMathematicalAgent._analyze_problem_node without duplication.
    
    Responsibilities:
    - Problem classification and complexity assessment
    - Mathematical context extraction
    - Analysis chain execution
    - State transition preparation
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for problem analysis results
        
    Raises:
        AgentError: If problem analysis fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._analyze_problem_node(state)
        
    except Exception as e:
        logger.error(f"Problem analysis node failed: {e}", exc_info=True)
        raise AgentError(f"Problem analysis failed: {str(e)}") from e


@log_function_call(logger)
async def reasoning_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Mathematical reasoning workflow node.
    
    This node extracts and reuses the existing reasoning logic
    from ReactMathematicalAgent._reasoning_node without duplication.
    
    Responsibilities:
    - Mathematical reasoning chain execution
    - Step-by-step reasoning tracking
    - Iteration count management
    - Context-aware mathematical analysis
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for reasoning results
        
    Raises:
        AgentError: If reasoning process fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._reasoning_node(state)
        
    except Exception as e:
        logger.error(f"Reasoning node failed: {e}", exc_info=True)
        raise AgentError(f"Mathematical reasoning failed: {str(e)}") from e


@log_function_call(logger)
async def tool_action_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Tool selection and execution workflow node.
    
    This node extracts and reuses the existing tool action logic
    from ReactMathematicalAgent._tool_action_node without duplication.
    
    Responsibilities:
    - BigTool semantic search for tool recommendations
    - Tool registry integration and selection
    - Mathematical tool execution with parameter extraction
    - Tool result processing and observation recording
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for tool execution results
        
    Raises:
        ToolError: If tool execution fails
        AgentError: If tool selection process fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._tool_action_node(state)
        
    except ToolError:
        # Re-raise tool errors as-is
        raise
    except Exception as e:
        logger.error(f"Tool action node failed: {e}", exc_info=True)
        raise AgentError(f"Tool action failed: {str(e)}") from e


@log_function_call(logger)
async def validation_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Result validation and reflection workflow node.
    
    This node extracts and reuses the existing validation logic
    from ReactMathematicalAgent._validation_node without duplication.
    
    Responsibilities:
    - Mathematical result validation using validation chains
    - Confidence score calculation
    - Solution completeness assessment
    - Quality assurance and error detection
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for validation results
        
    Raises:
        ValidationError: If validation process fails
        AgentError: If validation logic encounters errors
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._validation_node(state)
        
    except ValidationError:
        # Re-raise validation errors as-is
        raise
    except Exception as e:
        logger.error(f"Validation node failed: {e}", exc_info=True)
        raise AgentError(f"Result validation failed: {str(e)}") from e


@log_function_call(logger)
async def final_response_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Final response generation workflow node.
    
    This node extracts and reuses the existing response generation logic
    from ReactMathematicalAgent._final_response_node without duplication.
    
    Responsibilities:
    - Final answer synthesis and formatting
    - Response chain execution for clear explanations
    - Mathematical result presentation
    - Workflow completion status management
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for final response
        
    Raises:
        AgentError: If response generation fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._final_response_node(state)
        
    except Exception as e:
        logger.error(f"Final response node failed: {e}", exc_info=True)
        raise AgentError(f"Response generation failed: {str(e)}") from e


@log_function_call(logger)
async def error_recovery_node(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> Dict[str, Any]:
    """
    Error recovery and fallback workflow node.
    
    This node extracts and reuses the existing error recovery logic
    from ReactMathematicalAgent._error_recovery_node without duplication.
    
    Responsibilities:
    - Error analysis and categorization
    - Recovery strategy determination
    - Fallback mechanism execution
    - Error context preservation for debugging
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with initialized components
        
    Returns:
        Dict[str, Any]: State updates for error recovery
        
    Raises:
        AgentError: If error recovery fails completely
    """
    try:
        # REUSE existing implementation - zero duplication
        return await agent._error_recovery_node(state)
        
    except Exception as e:
        logger.error(f"Error recovery node failed: {e}", exc_info=True)
        raise AgentError(f"Error recovery failed: {str(e)}") from e


# === Node Registry and Factory Functions ===

class NodeRegistry:
    """
    Registry for workflow nodes following the Registry pattern.
    
    Provides centralized access to all workflow nodes with professional
    error handling and logging capabilities.
    """
    
    _nodes = {
        "analyze_problem": analyze_problem_node,
        "reasoning": reasoning_node,
        "tool_action": tool_action_node,
        "validation": validation_node,
        "final_response": final_response_node,
        "error_recovery": error_recovery_node,
    }
    
    @classmethod
    def get_node(cls, node_name: str):
        """
        Get workflow node by name.
        
        Args:
            node_name: Name of the workflow node
            
        Returns:
            Callable: Node function
            
        Raises:
            ValueError: If node name is not found
        """
        if node_name not in cls._nodes:
            available_nodes = list(cls._nodes.keys())
            raise ValueError(f"Node '{node_name}' not found. Available nodes: {available_nodes}")
        
        return cls._nodes[node_name]
    
    @classmethod
    def list_nodes(cls) -> List[str]:
        """
        List all available workflow nodes.
        
        Returns:
            List[str]: Names of all available nodes
        """
        return list(cls._nodes.keys())
    
    @classmethod
    def validate_nodes(cls) -> Dict[str, bool]:
        """
        Validate all nodes are properly callable.
        
        Returns:
            Dict[str, bool]: Validation status for each node
        """
        validation_results = {}
        for node_name, node_func in cls._nodes.items():
            try:
                # Check if function is callable and has correct signature
                import inspect
                signature = inspect.signature(node_func)
                params = list(signature.parameters.keys())
                
                is_valid = (
                    callable(node_func) and
                    len(params) >= 2 and
                    "state" in params and
                    "agent" in params
                )
                validation_results[node_name] = is_valid
                
            except Exception as e:
                logger.warning(f"Node validation failed for {node_name}: {e}")
                validation_results[node_name] = False
        
        return validation_results


# === Utility Functions ===

def create_node_wrapper(node_func, agent: "ReactMathematicalAgent"):
    """
    Create a wrapper function for a node that binds the agent.
    
    This follows the Partial Application pattern to create
    LangGraph-compatible node functions.
    
    Args:
        node_func: The node function to wrap
        agent: ReactMathematicalAgent instance
        
    Returns:
        Callable: Wrapped node function for LangGraph
    """
    async def wrapped_node(state: MathAgentState) -> Dict[str, Any]:
        """LangGraph-compatible node wrapper."""
        return await node_func(state, agent)
    
    # Preserve function metadata
    wrapped_node.__name__ = f"wrapped_{node_func.__name__}"
    wrapped_node.__doc__ = f"LangGraph wrapper for {node_func.__name__}"
    
    return wrapped_node


def create_all_node_wrappers(agent: "ReactMathematicalAgent") -> Dict[str, Any]:
    """
    Create all node wrappers for an agent.
    
    This factory function creates all LangGraph-compatible node wrappers
    for a ReactMathematicalAgent instance following the Factory pattern.
    
    Args:
        agent: ReactMathematicalAgent instance
        
    Returns:
        Dict[str, Any]: Dictionary of node name to wrapper function
    """
    node_wrappers = {}
    
    for node_name in NodeRegistry.list_nodes():
        node_func = NodeRegistry.get_node(node_name)
        node_wrappers[node_name] = create_node_wrapper(node_func, agent)
    
    logger.info(f"Created {len(node_wrappers)} node wrappers for agent")
    return node_wrappers


# === Module Validation ===

def validate_module() -> Dict[str, Any]:
    """
    Validate the nodes module for correctness.
    
    Returns:
        Dict[str, Any]: Validation results
    """
    return {
        "nodes_available": NodeRegistry.list_nodes(),
        "nodes_validation": NodeRegistry.validate_nodes(),
        "module_loaded": True,
        "total_nodes": len(NodeRegistry.list_nodes())
    }


# Export all public components
__all__ = [
    # Core node functions
    "analyze_problem_node",
    "reasoning_node", 
    "tool_action_node",
    "validation_node",
    "final_response_node",
    "error_recovery_node",
    
    # Registry and factory functions
    "NodeRegistry",
    "create_node_wrapper",
    "create_all_node_wrappers",
    
    # Utility functions
    "validate_module"
]


# Module initialization logging
logger.info(f"Workflow nodes module loaded with {len(NodeRegistry.list_nodes())} nodes")
