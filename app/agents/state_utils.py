"""State utilities for MathAgentState management.

This module provides utility functions for creating, validating, and managing
the MathAgentState, following professional patterns and ensuring consistency
with existing infrastructure.

Key features:
- State creation with sensible defaults
- Validation with comprehensive error checking
- Serialization for persistence and checkpointing
- Safe state updates with atomic operations
- Integration with existing AgentMemory
"""

import json
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from datetime import datetime
from copy import deepcopy

from ..core.logging import get_logger
from ..core.exceptions import ValidationError, StateError
from ..models.agent_state import AgentMemory
from .state import MathAgentState, WorkflowStatus, WorkflowSteps, get_empty_math_agent_state

logger = get_logger(__name__)


def create_initial_state(
    session_id: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[UUID] = None,
    max_iterations: int = 10,
    agent_config: Optional[Dict[str, Any]] = None
) -> MathAgentState:
    """
    Create initial state for a new conversation.
    
    This function creates a properly initialized MathAgentState with
    sensible defaults while allowing customization of key parameters.
    
    Args:
        session_id: Unique session identifier
        user_id: Optional user identifier
        conversation_id: Optional conversation ID (generates new if None)
        max_iterations: Maximum reasoning iterations allowed
        agent_config: Optional agent configuration
        
    Returns:
        MathAgentState: Fully initialized state
        
    Raises:
        ValidationError: If parameters are invalid
    """
    try:
        # Validation
        if not session_id or not isinstance(session_id, str):
            raise ValidationError("session_id must be a non-empty string")
            
        if max_iterations <= 0:
            raise ValidationError("max_iterations must be positive")
            
        # Start with empty state
        state = get_empty_math_agent_state()
        
        # Set provided values
        state["session_id"] = session_id
        state["user_id"] = user_id
        state["conversation_id"] = conversation_id or uuid4()
        state["max_iterations"] = max_iterations
        state["agent_config"] = agent_config or {}
        
        # Initialize with empty AgentMemory for consistency
        initial_memory = AgentMemory()
        state["agent_memory"] = _serialize_agent_memory(initial_memory)
        
        logger.info(
            f"Created initial state for session {session_id} "
            f"with conversation {state['conversation_id']}"
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Failed to create initial state: {e}")
        raise StateError(f"State creation failed: {e}") from e


def validate_state(state: MathAgentState) -> bool:
    """
    Validate MathAgentState structure and content.
    
    Performs comprehensive validation of the state to ensure
    it meets all requirements and constraints.
    
    Args:
        state: State to validate
        
    Returns:
        bool: True if valid
        
    Raises:
        ValidationError: If state is invalid
    """
    try:
        # Required fields validation
        required_fields = [
            "messages", "conversation_id", "session_id", "current_step",
            "iteration_count", "max_iterations", "workflow_status"
        ]
        
        for field in required_fields:
            if field not in state:
                raise ValidationError(f"Missing required field: {field}")
                
        # Type validations
        if not isinstance(state["messages"], list):
            raise ValidationError("messages must be a list")
            
        if not isinstance(state["conversation_id"], UUID):
            raise ValidationError("conversation_id must be a UUID")
            
        if not isinstance(state["session_id"], str):
            raise ValidationError("session_id must be a string")
            
        # Value validations
        if state["iteration_count"] < 0:
            raise ValidationError("iteration_count cannot be negative")
            
        if state["max_iterations"] <= 0:
            raise ValidationError("max_iterations must be positive")
            
        if state["iteration_count"] > state["max_iterations"]:
            raise ValidationError("iteration_count exceeds max_iterations")
            
        # Workflow status validation
        valid_statuses = [
            WorkflowStatus.ACTIVE,
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.PAUSED
        ]
        if state["workflow_status"] not in valid_statuses:
            raise ValidationError(f"Invalid workflow_status: {state['workflow_status']}")
            
        # Current step validation
        valid_steps = [
            WorkflowSteps.START,
            WorkflowSteps.REASONING,
            WorkflowSteps.TOOL_SELECTION,
            WorkflowSteps.TOOL_EXECUTION,
            WorkflowSteps.OBSERVATION,
            WorkflowSteps.REFLECTION,
            WorkflowSteps.ANSWER_GENERATION,
            WorkflowSteps.END
        ]
        if state["current_step"] not in valid_steps:
            raise ValidationError(f"Invalid current_step: {state['current_step']}")
            
        # List field validations
        list_fields = [
            "reasoning_steps", "available_tools", "selected_tools",
            "tool_calls", "tool_results", "intermediate_results",
            "visualizations", "solution_steps", "error_history",
            "thought_process", "observations"
        ]
        
        for field in list_fields:
            if not isinstance(state[field], list):
                raise ValidationError(f"{field} must be a list")
                
        # Dictionary field validations
        dict_fields = [
            "mathematical_context", "domain_knowledge", "agent_memory",
            "memory_context", "token_usage", "performance_metrics",
            "agent_config"
        ]
        
        for field in dict_fields:
            if not isinstance(state[field], dict):
                raise ValidationError(f"{field} must be a dictionary")
                
        logger.debug(f"State validation passed for conversation {state['conversation_id']}")
        return True
        
    except Exception as e:
        logger.error(f"State validation failed: {e}")
        raise ValidationError(f"Invalid state: {e}") from e


def serialize_state(state: MathAgentState) -> str:
    """
    Serialize MathAgentState to JSON string.
    
    Handles special types like UUID and datetime for persistence.
    
    Args:
        state: State to serialize
        
    Returns:
        str: JSON string representation
        
    Raises:
        StateError: If serialization fails
    """
    try:
        # Create a copy to avoid modifying original
        serializable_state = deepcopy(state)
        
        # Handle special types
        serializable_state["conversation_id"] = str(state["conversation_id"])
        serializable_state["created_at"] = state["created_at"].isoformat()
        serializable_state["updated_at"] = state["updated_at"].isoformat()
        
        # Serialize messages (simplified - in real implementation would handle BaseMessage)
        serializable_state["messages"] = _serialize_messages(state["messages"])
        
        return json.dumps(serializable_state, default=str, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to serialize state: {e}")
        raise StateError(f"Serialization failed: {e}") from e


def deserialize_state(state_json: str) -> MathAgentState:
    """
    Deserialize JSON string to MathAgentState.
    
    Reconstructs special types from their serialized forms.
    
    Args:
        state_json: JSON string to deserialize
        
    Returns:
        MathAgentState: Reconstructed state
        
    Raises:
        StateError: If deserialization fails
    """
    try:
        data = json.loads(state_json)
        
        # Reconstruct special types
        data["conversation_id"] = UUID(data["conversation_id"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        # Deserialize messages (simplified)
        data["messages"] = _deserialize_messages(data["messages"])
        
        # Create state from data
        state = MathAgentState(**data)
        
        # Validate reconstructed state
        validate_state(state)
        
        return state
        
    except Exception as e:
        logger.error(f"Failed to deserialize state: {e}")
        raise StateError(f"Deserialization failed: {e}") from e


def update_state_safely(
    state: MathAgentState,
    updates: Dict[str, Any],
    validate_after_update: bool = True
) -> MathAgentState:
    """
    Safely update state with validation and error handling.
    
    Provides atomic updates with rollback capability on validation failure.
    
    Args:
        state: Current state
        updates: Dictionary of updates to apply
        validate_after_update: Whether to validate after update
        
    Returns:
        MathAgentState: Updated state
        
    Raises:
        StateError: If update fails or validation fails
    """
    try:
        # Create backup for rollback
        original_state = deepcopy(state)
        
        # Apply updates
        updated_state = {**state, **updates}
        
        # Always update the timestamp
        updated_state["updated_at"] = datetime.now()
        
        # Validate if requested
        if validate_after_update:
            validate_state(updated_state)
            
        logger.debug(f"State updated successfully for conversation {state['conversation_id']}")
        return updated_state
        
    except Exception as e:
        logger.error(f"Failed to update state: {e}")
        # Rollback is automatic since we work with copies
        raise StateError(f"State update failed: {e}") from e


def _serialize_agent_memory(memory: AgentMemory) -> Dict[str, Any]:
    """
    Serialize AgentMemory for state storage.
    
    Reuses existing AgentMemory structure for consistency.
    
    Args:
        memory: AgentMemory instance
        
    Returns:
        Dict[str, Any]: Serialized memory
    """
    try:
        return {
            "short_term": memory.short_term,
            "long_term": memory.long_term,
            "context_window": memory.context_window,
            "max_context_size": memory.max_context_size
        }
    except Exception as e:
        logger.warning(f"Failed to serialize AgentMemory: {e}")
        return {}


def _deserialize_agent_memory(memory_data: Dict[str, Any]) -> AgentMemory:
    """
    Deserialize AgentMemory from state storage.
    
    Args:
        memory_data: Serialized memory data
        
    Returns:
        AgentMemory: Reconstructed memory instance
    """
    try:
        memory = AgentMemory()
        memory.short_term = memory_data.get("short_term", {})
        memory.long_term = memory_data.get("long_term", {})
        memory.context_window = memory_data.get("context_window", [])
        memory.max_context_size = memory_data.get("max_context_size", 50)
            
        return memory
    except Exception as e:
        logger.warning(f"Failed to deserialize AgentMemory: {e}")
        return AgentMemory()


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Serialize messages for storage.
    
    Simplified implementation - in production would handle BaseMessage properly.
    
    Args:
        messages: List of messages
        
    Returns:
        List[Dict[str, Any]]: Serialized messages
    """
    try:
        return [{"content": str(msg), "type": type(msg).__name__} for msg in messages]
    except Exception as e:
        logger.warning(f"Failed to serialize messages: {e}")
        return []


def _deserialize_messages(messages_data: List[Dict[str, Any]]) -> List[Any]:
    """
    Deserialize messages from storage.
    
    Simplified implementation - in production would reconstruct BaseMessage properly.
    
    Args:
        messages_data: Serialized messages
        
    Returns:
        List[Any]: Reconstructed messages
    """
    try:
        return [msg_data.get("content", "") for msg_data in messages_data]
    except Exception as e:
        logger.warning(f"Failed to deserialize messages: {e}")
        return []


def get_state_summary(state: MathAgentState) -> Dict[str, Any]:
    """
    Get a summary of the current state for logging/monitoring.
    
    Args:
        state: State to summarize
        
    Returns:
        Dict[str, Any]: State summary
    """
    return {
        "conversation_id": str(state["conversation_id"]),
        "session_id": state["session_id"],
        "current_step": state["current_step"],
        "iteration_count": state["iteration_count"],
        "max_iterations": state["max_iterations"],
        "workflow_status": state["workflow_status"],
        "error_count": state["error_count"],
        "tool_calls_count": len(state["tool_calls"]),
        "messages_count": len(state["messages"]),
        "has_final_answer": state["final_answer"] is not None,
        "execution_time": state["execution_time"],
        "updated_at": state["updated_at"].isoformat()
    }


def create_initial_state(
    problem: str,
    session_id: Optional[str] = None,
    context: Optional[List[str]] = None,
    **kwargs
) -> MathAgentState:
    """
    Create initial state for mathematical problem solving.
    
    Simple version for the unified architecture interface.
    
    Args:
        problem: Mathematical problem to solve
        session_id: Optional session identifier
        context: Optional context from previous interactions
        **kwargs: Additional parameters
        
    Returns:
        MathAgentState: Initial state for problem solving
    """
    from uuid import uuid4
    
    # Create basic initial state
    state = get_empty_math_agent_state()
    
    # Set required fields
    state.update({
        "current_problem": problem,
        "session_id": session_id or str(uuid4()),
        "conversation_id": uuid4(),
        "context": context or [],
        "current_step": WorkflowSteps.START,
        "workflow_status": WorkflowStatus.ACTIVE,
        "iteration_count": 0,
        "max_iterations": kwargs.get("max_iterations", 10),
        "confidence_score": 0.0,
        "reasoning_trace": [],
        "tools_to_use": [],
        "tool_results": [],
        "messages": []
    })
    
    return state


def format_agent_response(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format raw agent result for client consumption.
    
    Args:
        raw_result: Raw result from agent workflow
        
    Returns:
        Dict: Formatted response for client
    """
    try:
        # Extract key information
        final_answer = raw_result.get("final_answer", "")
        solution_steps = raw_result.get("solution_steps", [])
        explanation = raw_result.get("explanation", "")
        confidence = raw_result.get("confidence_score", 0.0)
        is_complete = raw_result.get("is_complete", False)
        status = raw_result.get("status", WorkflowStatus.ACTIVE)
        
        # Format response
        formatted = {
            "answer": final_answer,
            "steps": solution_steps,
            "explanation": explanation,
            "confidence": confidence,
            "success": is_complete and status == WorkflowStatus.COMPLETED,
            "status": status,
            "metadata": {
                "workflow_steps": raw_result.get("reasoning_trace", []),
                "tools_used": raw_result.get("tool_results", []),
                "iteration_count": raw_result.get("iteration_count", 0),
                "current_step": raw_result.get("current_step", "unknown")
            }
        }
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting agent response: {e}")
        return {
            "answer": "Error formatting response",
            "steps": [],
            "explanation": f"Error: {str(e)}",
            "confidence": 0.0,
            "success": False,
            "status": WorkflowStatus.FAILED,
            "metadata": {"error": str(e)}
        }
