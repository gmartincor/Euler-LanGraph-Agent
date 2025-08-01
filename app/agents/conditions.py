"""Professional LangGraph Conditional Edge Functions - Real Implementation.

This module contains the actual conditional edge logic using extracted
core business logic components, eliminating circular dependencies.

Key Design Principles Applied:
- Pure Functions: Conditions are stateless pure functions
- Single Responsibility: Each condition has one decision purpose
- Zero Circular Dependencies: No backwards references to agent classes
- Real Business Logic: Actual decision logic, not delegation wrappers
- Professional Error Handling: Robust condition evaluation

Architecture Benefits:
- No Code Duplication: Uses actual state analysis for decisions
- Testable: Each condition can be tested independently
- Maintainable: Clear decision logic without coupling
- Professional Quality: Production-ready decision functions
"""

from typing import Any, Dict, List, Optional

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError
from .state import MathAgentState, WorkflowSteps, WorkflowStatus

logger = get_logger(__name__)


# === Professional Conditional Edge Functions ===

@log_function_call(logger)
def should_use_tools(state: MathAgentState) -> str:
    """
    Determine if mathematical tools should be used in the workflow.
    
    This function implements actual decision logic based on state analysis,
    providing real business logic without circular dependencies.
    
    Args:
        state: Current mathematical agent state
        
    Returns:
        str: Next workflow step ("use_tools", "validate", or "error")
    """
    try:
        # Check for error conditions first
        error_count = state.get('error_count', 0)
        if error_count > 3:
            logger.warning(f"Too many errors ({error_count}), redirecting to error recovery")
            return "error"
        
        # Check if we have tools selected
        selected_tools = state.get('selected_tools', [])
        if not selected_tools:
            logger.info("No tools selected, need tool selection")
            return "select_tools"
        
        # Check if reasoning indicates tool usage is needed
        reasoning_chain = state.get('reasoning_chain', [])
        if reasoning_chain:
            latest_reasoning = reasoning_chain[-1]
            next_action = latest_reasoning.get('next_action', 'use_tools')
            
            if next_action == 'use_tools':
                logger.info("Reasoning indicates tools should be used")
                return "use_tools"
            elif next_action == 'finalize':
                logger.info("Reasoning indicates ready for validation")
                return "validate"
        
        # Check if we already have tool results that need validation
        tool_results = state.get('tool_results', {})
        if tool_results:
            logger.info("Tool results available, proceeding to validation")
            return "validate"
        
        # Default to using tools
        logger.info("Default decision: use tools")
        return "use_tools"
        
    except Exception as e:
        logger.error(f"Error in should_use_tools condition: {e}")
        return "error"


@log_function_call(logger)
def should_continue_reasoning(state: MathAgentState) -> str:
    """
    Determine if additional reasoning iterations are needed.
    
    Args:
        state: Current mathematical agent state
        
    Returns:
        str: Next workflow step ("continue", "finalize", or "error")
    """
    try:
        # Check iteration limits
        iteration_count = state.get('iteration_count', 0)
        max_iterations = state.get('max_iterations', 10)
        
        if iteration_count >= max_iterations:
            logger.warning(f"Maximum iterations ({max_iterations}) reached")
            return "finalize"
        
        # Check confidence score
        confidence_score = state.get('confidence_score', 0.0)
        if confidence_score > 0.8:
            logger.info(f"High confidence ({confidence_score:.2f}), ready to finalize")
            return "finalize"
        
        # Check if we have a final answer
        final_answer = state.get('final_answer')
        if final_answer is not None:
            logger.info("Final answer available, ready to finalize")
            return "finalize"
        
        # Check error conditions
        error_count = state.get('error_count', 0)
        if error_count > 2:
            logger.warning(f"Multiple errors ({error_count}), may need error recovery")
            return "error"
        
        # Continue reasoning if none of the above conditions are met
        logger.info("Continuing reasoning iteration")
        return "continue"
        
    except Exception as e:
        logger.error(f"Error in should_continue_reasoning condition: {e}")
        return "error"


@log_function_call(logger)
def should_finalize(state: MathAgentState) -> str:
    """
    Determine if the workflow should finalize with current results.
    
    Args:
        state: Current mathematical agent state
        
    Returns:
        str: Next workflow step ("finalize", "continue", or "error")
    """
    try:
        # Check if we have a valid final answer
        final_answer = state.get('final_answer')
        confidence_score = state.get('confidence_score', 0.0)
        
        if final_answer is not None and confidence_score > 0.6:
            logger.info(f"Ready to finalize (confidence: {confidence_score:.2f})")
            return "finalize"
        
        # Check if tool results are sufficient
        tool_results = state.get('tool_results', {})
        if tool_results and confidence_score > 0.5:
            # Check if results look complete
            result_quality = _assess_result_quality(tool_results)
            if result_quality > 0.7:
                logger.info(f"Tool results sufficient for finalization (quality: {result_quality:.2f})")
                return "finalize"
        
        # Check if we've reached maximum attempts
        iteration_count = state.get('iteration_count', 0)
        max_iterations = state.get('max_iterations', 10)
        
        if iteration_count >= max_iterations:
            logger.warning("Maximum iterations reached, forcing finalization")
            return "finalize"
        
        # Check for persistent errors
        error_count = state.get('error_count', 0)
        if error_count > 3:
            logger.warning(f"Too many errors ({error_count}), forcing finalization")
            return "error"
        
        # Continue if not ready to finalize
        logger.info("Not ready to finalize, continuing workflow")
        return "continue"
        
    except Exception as e:
        logger.error(f"Error in should_finalize condition: {e}")
        return "error"


@log_function_call(logger)
def should_retry(state: MathAgentState) -> str:
    """
    Determine if the workflow should retry after an error.
    
    Args:
        state: Current mathematical agent state
        
    Returns:
        str: Next workflow step ("retry", "recover", or "fail")
    """
    try:
        error_count = state.get('error_count', 0)
        max_retries = 3
        
        # Check if we've exceeded retry limit
        if error_count >= max_retries:
            logger.warning(f"Maximum retries ({max_retries}) exceeded")
            return "fail"
        
        # Analyze the type of error
        last_error = state.get('last_error', '')
        
        # Temporary/recoverable errors - retry
        recoverable_error_patterns = [
            'timeout',
            'connection',
            'temporary',
            'network',
            'unavailable'
        ]
        
        if any(pattern in last_error.lower() for pattern in recoverable_error_patterns):
            logger.info(f"Recoverable error detected, retrying (attempt {error_count + 1})")
            return "retry"
        
        # Tool execution errors - try recovery
        if 'tool' in last_error.lower() or 'execution' in last_error.lower():
            logger.info("Tool execution error, attempting recovery")
            return "recover"
        
        # Validation errors - try different approach
        if 'validation' in last_error.lower() or 'invalid' in last_error.lower():
            logger.info("Validation error, attempting recovery with different approach")
            return "recover"
        
        # Default to recovery for first few attempts
        if error_count < 2:
            logger.info(f"Attempting error recovery (attempt {error_count + 1})")
            return "recover"
        
        # Too many errors - fail
        logger.warning("Too many errors, failing workflow")
        return "fail"
        
    except Exception as e:
        logger.error(f"Error in should_retry condition: {e}")
        return "fail"


@log_function_call(logger)
def workflow_complete(state: MathAgentState) -> bool:
    """
    Check if the workflow has completed successfully.
    
    Args:
        state: Current mathematical agent state
        
    Returns:
        bool: True if workflow is complete, False otherwise
    """
    try:
        workflow_status = state.get('workflow_status', WorkflowStatus.ACTIVE)
        
        # Check explicit completion status
        if workflow_status in [WorkflowStatus.COMPLETED, WorkflowStatus.COMPLETED_WITH_ERRORS]:
            return True
        
        # Check if we have a final answer
        final_answer = state.get('final_answer')
        if final_answer is not None:
            return True
        
        # Check if maximum iterations reached
        iteration_count = state.get('iteration_count', 0)
        max_iterations = state.get('max_iterations', 10)
        
        if iteration_count >= max_iterations:
            logger.info("Maximum iterations reached, considering workflow complete")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error in workflow_complete condition: {e}")
        return True  # Fail-safe: consider complete to avoid infinite loops


# === Private Helper Functions ===

def _assess_result_quality(tool_results: Dict[str, Any]) -> float:
    """
    Assess the quality of tool results for decision making.
    
    Args:
        tool_results: Dictionary of tool execution results
        
    Returns:
        float: Quality score between 0 and 1
    """
    if not tool_results:
        return 0.0
    
    quality_score = 0.0
    total_tools = len(tool_results)
    
    for tool_name, result in tool_results.items():
        tool_quality = 0.0
        
        # Check if result exists and is not None
        if result is not None:
            tool_quality += 0.3
        
        # Check if result has expected structure
        if isinstance(result, dict):
            tool_quality += 0.2
            
            # Check for success indicators
            if result.get('success', False):
                tool_quality += 0.3
            
            # Check for actual result data
            if 'result' in result or 'data' in result:
                tool_quality += 0.2
        
        # Check for error indicators
        if isinstance(result, dict) and result.get('error'):
            tool_quality = max(0.0, tool_quality - 0.4)
        
        quality_score += tool_quality
    
    # Average quality across all tools
    return min(1.0, quality_score / total_tools)
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance (for context)
        
    Returns:
        str: Decision for next workflow step ("use_tools", "validate", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # Check for error conditions first
        error_count = state.get("error_count", 0)
        if error_count > 0:
            return "error"
        
        # Check if reasoning suggests tool usage
        reasoning_steps = state.get("reasoning_steps", [])
        current_reasoning = state.get("current_reasoning", "")
        
        # Combine all reasoning text for analysis
        all_reasoning = ""
        if reasoning_steps:
            all_reasoning += " ".join(reasoning_steps)
        if current_reasoning:
            all_reasoning += " " + current_reasoning
            
        if all_reasoning:
            reasoning_lower = all_reasoning.lower()
            
            # Look for mathematical patterns that need tools
            tool_indicators = [
                "calculate", "compute", "integral", "derivative", 
                "plot", "graph", "solve", "evaluate", "tool"
            ]
            
            if any(indicator in reasoning_lower for indicator in tool_indicators):
                return "use_tools"
        
        # Default to validation if no tool usage indicated
        return "validate"
        
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
    
    This function implements the core decision logic for reasoning continuation,
    centralizing the logic that was previously in ReactMathematicalAgent.
    
    Decision Logic:
    - "error": If error conditions are detected
    - "continue": If within iteration limits and reasoning incomplete
    - "validate": If reasoning is complete or max iterations reached
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance (for context)
        
    Returns:
        str: Decision for next workflow step ("continue", "validate", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # Check for error conditions first
        error_count = state.get("error_count", 0)
        if error_count > 0:
            return "error"
        
        # Check iteration limits
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        
        if iteration_count >= max_iterations:
            return "validate"
        
        # Check if reasoning is complete (simple heuristic)
        reasoning_steps = state.get("reasoning_steps", [])
        if reasoning_steps:
            last_step = reasoning_steps[-1].lower()
            completion_indicators = ["answer", "result", "solution", "final"]
            
            if any(indicator in last_step for indicator in completion_indicators):
                return "validate"
        
        # Continue reasoning if not complete and under limits
        return "continue"
        
    except Exception as e:
        logger.error(f"Reasoning continuation decision failed: {e}", exc_info=True)
        return "error"



@log_function_call(logger)
def should_finalize(
    state: MathAgentState, 
    agent: "ReactMathematicalAgent"
) -> str:
    """
    Determine if the mathematical solution should be finalized.
    
    This function implements the core decision logic for solution finalization,
    centralizing the logic that was previously in ReactMathematicalAgent.
    
    Decision Logic:
    - "finalize": If confidence score is high enough
    - "continue": If solution needs more work
    - "error": If validation indicates errors
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance (for context)
        
    Returns:
        str: Decision for next workflow step ("finalize", "continue", "error")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # Check for error conditions first
        error_count = state.get("error_count", 0)
        if error_count > 0:
            return "error"
        
        # Check confidence score
        confidence_score = state.get("confidence_score", 0.0)
        if confidence_score >= 0.8:  # High confidence threshold
            return "finalize"
        
        # Check if we have valid tool results
        tool_results = state.get("tool_results", [])
        if tool_results:
            # Check if last tool result was successful
            last_result = tool_results[-1] if tool_results else None
            if last_result and last_result.get("success"):
                return "finalize"
        
        # Check iteration limits to prevent infinite loops
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        
        if iteration_count >= max_iterations:
            return "finalize"  # Force finalization to prevent loops
        
        # Default to continue working
        return "continue"
        
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
    
    This function implements the core decision logic for retry behavior,
    centralizing the logic that was previously in ReactMathematicalAgent.
    
    Decision Logic:
    - "retry": If error count is within limits and recovery possible
    - "finalize": If retry limits exceeded or no recovery possible
    
    Args:
        state: Current mathematical agent state
        agent: ReactMathematicalAgent instance (for context)
        
    Returns:
        str: Decision for next workflow step ("retry", "finalize")
        
    Raises:
        AgentError: If decision logic fails
    """
    try:
        # Check retry limits
        error_count = state.get("error_count", 0)
        max_retries = state.get("max_retries", 3)
        
        if error_count >= max_retries:
            return "finalize"  # Exceeded retry limits
        
        # Check if error is recoverable
        last_error = state.get("last_error", "")
        if last_error:
            # Non-recoverable errors
            non_recoverable = [
                "authentication", "permission", "network", 
                "quota_exceeded", "api_key_invalid"
            ]
            
            if any(error_type in last_error.lower() for error_type in non_recoverable):
                return "finalize"
        
        # Default to retry if within limits and recoverable
        return "retry"
        
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
