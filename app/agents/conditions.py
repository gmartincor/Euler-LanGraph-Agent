"""Professional LangGraph Conditional Edge Functions - Simplified.

Simple conditional functions for the mathematical workflow.
"""

from typing import Any, Dict
from ..core.logging import get_logger
from .state import MathAgentState, WorkflowSteps, WorkflowStatus

logger = get_logger(__name__)


def should_continue_reasoning(state: MathAgentState) -> str:
    """Determine if reasoning should continue."""
    try:
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        analysis = state.get('problem_analysis', {})
        if analysis.get('complexity') == 'high':
            return "continue"
        else:
            return "continue"  # Always continue to reasoning for now
            
    except Exception as e:
        logger.error(f"Error in should_continue_reasoning: {e}")
        return "error"


def should_execute_tools(state: MathAgentState) -> str:
    """Determine if tools should be executed."""
    try:
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        reasoning_result = state.get('reasoning_result', {})
        tools_needed = reasoning_result.get('tools_needed', [])
        
        if tools_needed:
            return "execute_tools"
        else:
            return "validate"
            
    except Exception as e:
        logger.error(f"Error in should_execute_tools: {e}")
        return "error"


def should_validate_result(state: MathAgentState) -> str:
    """Determine if results should be validated."""
    try:
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
    """Determine if the workflow should finalize."""
    try:
        if state.get('current_step') == WorkflowSteps.ERROR_RECOVERY:
            return "error"
        
        validation_result = state.get('validation_result', {})
        is_valid = validation_result.get('is_valid', False)
        
        if is_valid:
            return "finalize"
        else:
            return "retry"
            
    except Exception as e:
        logger.error(f"Error in should_finalize: {e}")
        return "error"
