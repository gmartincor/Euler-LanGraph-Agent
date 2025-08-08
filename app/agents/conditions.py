from typing import Any, Dict
from ..core.logging import get_logger
from .state import MathAgentState, WorkflowSteps, WorkflowStatus

logger = get_logger(__name__)


def _check_loop_limit(state: MathAgentState) -> bool:
    """Check if workflow has exceeded iteration limit."""
    iteration_count = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 10)
    
    if iteration_count >= max_iterations:
        logger.warning(f"Loop limit reached: {iteration_count}/{max_iterations}")
        return True
    return False


def should_continue_reasoning(state: MathAgentState) -> str:
    """Determine if reasoning should continue with loop detection."""
    try:
        if _check_loop_limit(state):
            logger.warning("Forcing finalization due to iteration limit")
            return "error"
        
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        analysis = state.get('problem_analysis', {})
        if analysis.get('complexity') == 'high':
            return "continue"
        else:
            return "continue"  
            
    except Exception as e:
        logger.error(f"Error in should_continue_reasoning: {e}")
        return "error"


def should_execute_tools(state: MathAgentState) -> str:
    """Determine if tools should be executed with loop detection."""
    try:
        if _check_loop_limit(state):
            logger.warning("Forcing finalization due to iteration limit")
            return "error"
        
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        tools_needed = state.get('tools_to_use', [])
        
        if tools_needed:
            logger.info(f"Tools identified for execution: {tools_needed}")
            return "execute_tools"
        else:
            # PROFESSIONAL PATTERN: Provide clear diagnostic information
            reasoning_result = state.get('reasoning_result', {})
            reasoning_tools = reasoning_result.get('tools_needed', [])
            
            if reasoning_tools:
                logger.error("CRITICAL: State management error detected")
                logger.error(f"reasoning_result.tools_needed: {reasoning_tools}")
                logger.error(f"state.tools_to_use: {tools_needed}")
                return "error"
            
            logger.info("No tools needed, proceeding to validation")
            return "validate"
            
    except Exception as e:
        logger.error(f"Error in should_execute_tools: {e}")
        return "error"


def should_validate_result(state: MathAgentState) -> str:
    """Determine if results should be validated with loop detection."""
    try:
        if _check_loop_limit(state):
            logger.warning("Forcing finalization due to iteration limit")
            return "error"
        
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        tool_results = state.get('tool_results', [])
        
        if not tool_results and state.get('tools_to_use'):
            return "retry"
        
        tool_errors = [r for r in tool_results if 'error' in r]
        if tool_errors and len(tool_errors) == len(tool_results):
            return "retry"
        
        return "validate"
        
    except Exception as e:
        logger.error(f"Error in should_validate_result: {e}")
        return "error"


def should_finalize(state: MathAgentState) -> str:
    """Determine if the workflow should finalize with loop detection."""
    try:
        if _check_loop_limit(state):
            logger.warning("Forcing finalization due to iteration limit")
            return "finalize"
        
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        validation_result = state.get('validation_result', {})
        is_valid = validation_result.get('is_valid', False)
        
        retry_count = state.get('iteration_count', 0)
        max_retries = state.get('max_iterations', 10) // 2
        
        if is_valid:
            return "finalize"
        elif retry_count >= max_retries:
            logger.warning(f"Max retries ({max_retries}) reached, forcing finalization")
            return "finalize"
        else:
            return "retry"
            
    except Exception as e:
        logger.error(f"Error in should_finalize: {e}")
        return "error"
