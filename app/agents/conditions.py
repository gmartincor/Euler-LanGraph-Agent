"""LangGraph Conditional Edge Functions for Mathematical ReAct Agent.

This module contains the extracted conditional edge logic from ReactMathematicalAgent,
following professional design patterns and DRY principles. Each condition function
determines workflow transitions while reusing existing tested implementations.

Key Design Patterns Applied:
- Strategy Pattern: Each condition implements a decision strategy
- Dependency Injection: Agent dependencies injected as parameters
- Single Responsibility: Each condition has one decision purpose
- Professional Error Handling: Robust condition evaluation
- State Pattern: Conditions evaluate current state for transitions

Architecture Benefits:
- Zero Code Duplication: Reuses existing ReactMathematicalAgent logic
- Enhanced Debuggability: Standalone conditions for individual testing
- Improved Workflow Control: Clear transition decision points
- Professional Quality: Maintains existing tested behavior
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError
from .state import MathAgentState

# Type checking imports to avoid circular dependencies
if TYPE_CHECKING:
    from .react_agent import ReactMathematicalAgent

logger = get_logger(__name__)


# === Core Conditional Edge Functions ===

@log_function_call(logger)
def should_use_tools(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> str:
    """
    Determine if mathematical tools should be used in the workflow.
    
    This function extracts and reuses the existing tool usage decision logic
    from ReactMathematicalAgent._should_use_tools without duplication.
    
    Decision Logic:
    - "error": If error conditions are detected
    - "use_tools": If reasoning indicates tool usage needed
    - "validate": If ready for result validation
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with decision logic
        
    Returns:
        str: Decision for next workflow step ("use_tools", "validate", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return agent._should_use_tools(state)
        
    except Exception as e:
        logger.error(f"Tool usage decision failed: {e}", exc_info=True)
        # Fallback to error recovery
        return "error"


@log_function_call(logger)
def should_continue_reasoning(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> str:
    """
    Determine if mathematical reasoning should continue.
    
    This function extracts and reuses the existing reasoning continuation logic
    from ReactMathematicalAgent._should_continue_reasoning without duplication.
    
    Decision Logic:
    - "error": If error conditions are detected
    - "continue": If within iteration limits and reasoning incomplete
    - "validate": If reasoning is complete or max iterations reached
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with decision logic
        
    Returns:
        str: Decision for next workflow step ("continue", "validate", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return agent._should_continue_reasoning(state)
        
    except Exception as e:
        logger.error(f"Reasoning continuation decision failed: {e}", exc_info=True)
        # Fallback to validation to prevent infinite loops
        return "validate"


@log_function_call(logger)
def should_finalize(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> str:
    """
    Determine if the mathematical solution should be finalized.
    
    This function extracts and reuses the existing finalization decision logic
    from ReactMathematicalAgent._should_finalize without duplication.
    
    Decision Logic:
    - "finalize": If confidence score is high enough
    - "continue": If solution needs more work
    - "error": If validation indicates errors
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with decision logic
        
    Returns:
        str: Decision for next workflow step ("finalize", "continue", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return agent._should_finalize(state)
        
    except Exception as e:
        logger.error(f"Finalization decision failed: {e}", exc_info=True)
        # Conservative fallback to continue working
        return "continue"


@log_function_call(logger)
def should_retry(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> str:
    """
    Determine if error recovery should retry or give up.
    
    This function extracts and reuses the existing retry decision logic
    from ReactMathematicalAgent._should_retry without duplication.
    
    Decision Logic:
    - "retry": If error count is within limits and recovery possible
    - "finalize": If max retries reached or recovery impossible
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance with decision logic
        
    Returns:
        str: Decision for next workflow step ("retry", "finalize")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # REUSE existing implementation - zero duplication
        return agent._should_retry(state)
        
    except Exception as e:
        logger.error(f"Retry decision failed: {e}", exc_info=True)
        # Conservative fallback to finalize to prevent infinite retries
        return "finalize"


# === Advanced Conditional Functions ===

@log_function_call(logger)
def needs_human_intervention(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> bool:
    """
    Determine if human intervention is needed.
    
    This is an extension condition for complex scenarios requiring
    human guidance or validation.
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance
        
    Returns:
        bool: True if human intervention needed
    """
    try:
        # Check for critical error conditions
        error_count = state.get("error_count", 0)
        max_iterations = state.get("max_iterations", 10)
        current_iterations = state.get("iteration_count", 0)
        
        # Human intervention criteria
        too_many_errors = error_count > 3
        max_iterations_exceeded = current_iterations >= max_iterations * 1.5
        critical_failure = state.get("workflow_status") == "critical_failure"
        
        return too_many_errors or max_iterations_exceeded or critical_failure
        
    except Exception as e:
        logger.warning(f"Human intervention check failed: {e}")
        return False  # Conservative default


@log_function_call(logger)
def is_problem_complex(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> bool:
    """
    Determine if the mathematical problem is complex.
    
    This condition helps determine if additional resources or
    specialized handling is needed.
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance
        
    Returns:
        bool: True if problem is complex
    """
    try:
        problem_complexity = state.get("problem_complexity", "medium")
        tool_count = len(state.get("tool_calls", []))
        reasoning_steps = len(state.get("reasoning_steps", []))
        
        # Complexity indicators
        high_complexity = problem_complexity in ["high", "very_high"]
        many_tools_used = tool_count > 5
        extensive_reasoning = reasoning_steps > 10
        
        return high_complexity or many_tools_used or extensive_reasoning
        
    except Exception as e:
        logger.warning(f"Problem complexity check failed: {e}")
        return False  # Conservative default


# === Conditional Edge Registry ===

class ConditionRegistry:
    """
    Registry for conditional edge functions following the Registry pattern.
    
    Provides centralized access to all workflow conditions with professional
    error handling and logging capabilities.
    """
    
    _conditions = {
        # Core workflow conditions
        "should_use_tools": should_use_tools,
        "should_continue_reasoning": should_continue_reasoning,
        "should_finalize": should_finalize,
        "should_retry": should_retry,
        
        # Advanced conditions
        "needs_human_intervention": needs_human_intervention,
        "is_problem_complex": is_problem_complex,
    }
    
    @classmethod
    def get_condition(cls, condition_name: str):
        """
        Get conditional function by name.
        
        Args:
            condition_name: Name of the conditional function
            
        Returns:
            Callable: Condition function
            
        Raises:
            ValueError: If condition name is not found
        """
        if condition_name not in cls._conditions:
            available_conditions = list(cls._conditions.keys())
            raise ValueError(f"Condition '{condition_name}' not found. "
                           f"Available conditions: {available_conditions}")
        
        return cls._conditions[condition_name]
    
    @classmethod
    def list_conditions(cls) -> List[str]:
        """
        List all available conditional functions.
        
        Returns:
            List[str]: Names of all available conditions
        """
        return list(cls._conditions.keys())
    
    @classmethod
    def validate_conditions(cls) -> Dict[str, bool]:
        """
        Validate all conditions are properly callable.
        
        Returns:
            Dict[str, bool]: Validation status for each condition
        """
        validation_results = {}
        for condition_name, condition_func in cls._conditions.items():
            try:
                # Check if function is callable and has correct signature
                import inspect
                signature = inspect.signature(condition_func)
                params = list(signature.parameters.keys())
                
                is_valid = (
                    callable(condition_func) and
                    len(params) >= 2 and
                    "state" in params and
                    "agent" in params
                )
                validation_results[condition_name] = is_valid
                
            except Exception as e:
                logger.warning(f"Condition validation failed for {condition_name}: {e}")
                validation_results[condition_name] = False
        
        return validation_results


# === Utility Functions ===

def create_condition_wrapper(condition_func, agent: "ReactMathematicalAgent"):
    """
    Create a wrapper function for a condition that binds the agent.
    
    This follows the Partial Application pattern to create
    LangGraph-compatible condition functions.
    
    Args:
        condition_func: The condition function to wrap
        agent: ReactMathematicalAgent instance
        
    Returns:
        Callable: Wrapped condition function for LangGraph
    """
    def wrapped_condition(state: MathAgentState):
        """LangGraph-compatible condition wrapper."""
        return condition_func(state, agent)
    
    # Preserve function metadata
    wrapped_condition.__name__ = f"wrapped_{condition_func.__name__}"
    wrapped_condition.__doc__ = f"LangGraph wrapper for {condition_func.__name__}"
    
    return wrapped_condition


def create_all_condition_wrappers(agent: "ReactMathematicalAgent") -> Dict[str, Any]:
    """
    Create all condition wrappers for an agent.
    
    This factory function creates all LangGraph-compatible condition wrappers
    for a ReactMathematicalAgent instance following the Factory pattern.
    
    Args:
        agent: ReactMathematicalAgent instance
        
    Returns:
        Dict[str, Any]: Dictionary of condition name to wrapper function
    """
    condition_wrappers = {}
    
    for condition_name in ConditionRegistry.list_conditions():
        condition_func = ConditionRegistry.get_condition(condition_name)
        condition_wrappers[condition_name] = create_condition_wrapper(condition_func, agent)
    
    logger.info(f"Created {len(condition_wrappers)} condition wrappers for agent")
    return condition_wrappers


# === Edge Configuration Helpers ===

def get_standard_edge_mappings() -> Dict[str, Dict[str, str]]:
    """
    Get standard edge mappings for common workflow transitions.
    
    Returns:
        Dict[str, Dict[str, str]]: Standard edge configurations
    """
    return {
        "reasoning_transitions": {
            "use_tools": "tool_action",
            "validate": "validation", 
            "error": "error_recovery"
        },
        "tool_action_transitions": {
            "continue": "reasoning",
            "validate": "validation",
            "error": "error_recovery"
        },
        "validation_transitions": {
            "finalize": "final_response",
            "continue": "reasoning",
            "error": "error_recovery"
        },
        "error_recovery_transitions": {
            "retry": "reasoning",
            "finalize": "final_response"
        }
    }


def create_conditional_edges_config(agent: "ReactMathematicalAgent") -> Dict[str, Any]:
    """
    Create complete conditional edges configuration for LangGraph.
    
    Args:
        agent: ReactMathematicalAgent instance
        
    Returns:
        Dict[str, Any]: Complete conditional edges configuration
    """
    condition_wrappers = create_all_condition_wrappers(agent)
    edge_mappings = get_standard_edge_mappings()
    
    return {
        "conditions": condition_wrappers,
        "mappings": edge_mappings,
        "agent_reference": agent
    }


# === Module Validation ===

def validate_module() -> Dict[str, Any]:
    """
    Validate the conditions module for correctness.
    
    Returns:
        Dict[str, Any]: Validation results
    """
    return {
        "conditions_available": ConditionRegistry.list_conditions(),
        "conditions_validation": ConditionRegistry.validate_conditions(),
        "edge_mappings": get_standard_edge_mappings(),
        "module_loaded": True,
        "total_conditions": len(ConditionRegistry.list_conditions())
    }


# Export all public components
__all__ = [
    # Core condition functions
    "should_use_tools",
    "should_continue_reasoning", 
    "should_finalize",
    "should_retry",
    
    # Advanced condition functions
    "needs_human_intervention",
    "is_problem_complex",
    
    # Registry and factory functions
    "ConditionRegistry",
    "create_condition_wrapper",
    "create_all_condition_wrappers",
    
    # Configuration helpers
    "get_standard_edge_mappings",
    "create_conditional_edges_config",
    
    # Utility functions
    "validate_module"
]


# Module initialization logging
logger.info(f"Workflow conditions module loaded with {len(ConditionRegistry.list_conditions())} conditions")
