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
