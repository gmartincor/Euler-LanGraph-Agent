"""Agent module for ReAct Mathematical Agent.

This module contains the LangGraph-based ReAct agent implementation
with state management and mathematical reasoning capabilities.
"""

# Import protection to avoid circular imports and missing dependencies
try:
    from .state import MathAgentState, WorkflowStatus, WorkflowSteps
    from .state_utils import (
        create_initial_state,
        validate_state,
        serialize_state,
        deserialize_state,
        update_state_safely,
        get_state_summary
    )
    
    __all__ = [
        "MathAgentState",
        "WorkflowStatus",
        "WorkflowSteps",
        "create_initial_state", 
        "validate_state",
        "serialize_state",
        "deserialize_state",
        "update_state_safely",
        "get_state_summary"
    ]
    
except ImportError as e:
    # Graceful degradation if dependencies are not available
    import logging
    logging.warning(f"Some agent dependencies not available: {e}")
    
    __all__ = []
