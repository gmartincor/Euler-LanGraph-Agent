"""Unit tests for agent state management.

This module tests the MathAgentState and related utilities        # Invalid max_iterations - should raise StateError 
        with pytest.raises(StateError):
            create_initial_state(problem="test", max_iterations=0) ensure
robust state management following the established patterns.
"""

import pytest
from uuid import UUID, uuid4
from datetime import datetime
from typing import Dict, Any

from app.agents.state import MathAgentState, WorkflowStatus, WorkflowSteps, get_empty_math_agent_state
from app.agents.state_utils import (
    create_initial_state,
    validate_state,
    serialize_state,
    deserialize_state,
    update_state_safely,
    get_state_summary
)
from app.core.exceptions import ValidationError, StateError


class TestMathAgentState:
    """Test MathAgentState creation and basic operations."""
    
    def test_get_empty_math_agent_state(self):
        """Test creation of empty state with proper defaults."""
        state = get_empty_math_agent_state()
        
        # Check required fields exist
        assert "messages" in state
        assert "conversation_id" in state
        assert "session_id" in state
        assert "current_step" in state
        assert "workflow_status" in state
        
        # Check default values
        assert state["messages"] == []
        assert isinstance(state["conversation_id"], UUID)
        assert state["current_step"] == WorkflowSteps.START
        assert state["workflow_status"] == WorkflowStatus.ACTIVE
        assert state["iteration_count"] == 0
        assert state["max_iterations"] == 10
        
        # Check timestamps
        assert isinstance(state["created_at"], datetime)
        assert isinstance(state["updated_at"], datetime)
        
        # Check collections are empty
        assert state["reasoning_steps"] == []
        assert state["tool_calls"] == []
        assert state["error_history"] == []
        
    def test_workflow_constants(self):
        """Test workflow constants are properly defined."""
        # Test WorkflowStatus constants
        assert hasattr(WorkflowStatus, 'ACTIVE')
        assert hasattr(WorkflowStatus, 'COMPLETED')
        assert hasattr(WorkflowStatus, 'FAILED')
        assert hasattr(WorkflowStatus, 'PAUSED')
        
        # Test WorkflowSteps constants
        assert hasattr(WorkflowSteps, 'START')
        assert hasattr(WorkflowSteps, 'REASONING')
        assert hasattr(WorkflowSteps, 'TOOL_SELECTION')
        assert hasattr(WorkflowSteps, 'END')


class TestStateUtilities:
    """Test state utility functions."""
    
    def test_create_initial_state_success(self):
        """Test successful creation of initial state."""
        session_id = "test-session-123"
        user_id = "user-456"
        conversation_id = uuid4()
        max_iterations = 15
        problem = "Test problem: x^2 + 1"
        
        state = create_initial_state(
            problem=problem,
            session_id=session_id,
            user_id=user_id,
            conversation_id=conversation_id,
            max_iterations=max_iterations
        )
        
        assert state["current_problem"] == problem  
        assert state["session_id"] == session_id
        assert state["user_id"] == user_id
        assert state["conversation_id"] == conversation_id
        assert state["max_iterations"] == max_iterations
        assert state["workflow_status"] == WorkflowStatus.ACTIVE
        
    def test_create_initial_state_validation_errors(self):
        """Test validation errors in initial state creation."""
        # Empty problem - should raise StateError (which wraps ValidationError)
        with pytest.raises(StateError):
            create_initial_state(problem="")
            
        # Invalid max_iterations - should raise StateError 
        with pytest.raises(StateError):
            create_initial_state(problem="test problem", session_id="test", max_iterations=0)
            
        with pytest.raises(StateError):
            create_initial_state(problem="test problem", session_id="test", max_iterations=-1)
    
    def test_validate_state_success(self):
        """Test successful state validation."""
        state = create_initial_state("test-session")
        
        # Should pass validation
        assert validate_state(state) is True
        
    def test_validate_state_missing_required_fields(self):
        """Test validation with missing required fields."""
        state = get_empty_math_agent_state()
        
        # Remove required field
        del state["conversation_id"]
        
        with pytest.raises(ValidationError, match="Missing required field"):
            validate_state(state)
            
    def test_validate_state_invalid_types(self):
        """Test validation with invalid field types."""
        state = create_initial_state("test-session")
        
        # Invalid type for messages
        state["messages"] = "not a list"
        
        with pytest.raises(ValidationError, match="messages must be a list"):
            validate_state(state)
            
    def test_validate_state_invalid_values(self):
        """Test validation with invalid field values."""
        state = create_initial_state("test-session")
        
        # Negative iteration count
        state["iteration_count"] = -1
        
        with pytest.raises(ValidationError, match="iteration_count cannot be negative"):
            validate_state(state)
            
        # Iteration count exceeds max
        state["iteration_count"] = 15
        state["max_iterations"] = 10
        
        with pytest.raises(ValidationError, match="iteration_count exceeds max_iterations"):
            validate_state(state)
            
        # Invalid workflow status
        state["iteration_count"] = 5  # Fix previous issue
        state["workflow_status"] = "invalid_status"
        
        with pytest.raises(ValidationError, match="Invalid workflow_status"):
            validate_state(state)
    
    def test_update_state_safely_success(self):
        """Test successful safe state updates."""
        original_state = create_initial_state("test-session")
        original_updated_at = original_state["updated_at"]
        
        updates = {
            "current_step": WorkflowSteps.REASONING,
            "iteration_count": 1,
            "current_reasoning": "Testing reasoning process"
        }
        
        updated_state = update_state_safely(original_state, updates)
        
        # Check updates were applied
        assert updated_state["current_step"] == WorkflowSteps.REASONING
        assert updated_state["iteration_count"] == 1
        assert updated_state["current_reasoning"] == "Testing reasoning process"
        
        # Check timestamp was updated
        assert updated_state["updated_at"] > original_updated_at
        
        # Check original state unchanged (immutable update)
        assert original_state["current_step"] == WorkflowSteps.START
        assert original_state["iteration_count"] == 0
    
    def test_update_state_safely_validation_error(self):
        """Test safe update with validation error."""
        state = create_initial_state("test-session")
        
        # Update that would cause validation error
        invalid_updates = {
            "iteration_count": -5
        }
        
        with pytest.raises(StateError):
            update_state_safely(state, invalid_updates)
            
        # Original state should be unchanged
        assert state["iteration_count"] == 0
    
    def test_serialize_deserialize_state_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        original_state = create_initial_state("test-session", user_id="test-user")
        
        # Add some data to make it more interesting
        original_state["reasoning_steps"] = ["Step 1", "Step 2"]
        original_state["current_reasoning"] = "Test reasoning"
        original_state["tool_calls"] = [{"tool": "test_tool", "args": {}}]
        
        # Serialize
        state_json = serialize_state(original_state)
        assert isinstance(state_json, str)
        
        # Deserialize
        deserialized_state = deserialize_state(state_json)
        
        # Check key fields match
        assert deserialized_state["session_id"] == original_state["session_id"]
        assert deserialized_state["user_id"] == original_state["user_id"]
        assert deserialized_state["conversation_id"] == original_state["conversation_id"]
        assert deserialized_state["reasoning_steps"] == original_state["reasoning_steps"]
        assert deserialized_state["current_reasoning"] == original_state["current_reasoning"]
        
        # Validate deserialized state
        assert validate_state(deserialized_state) is True
    
    def test_get_state_summary(self):
        """Test state summary generation."""
        state = create_initial_state("test-session")
        
        # Add some data
        state["current_step"] = WorkflowSteps.REASONING
        state["iteration_count"] = 3
        state["error_count"] = 1
        state["tool_calls"] = [{"tool": "test1"}, {"tool": "test2"}]
        state["final_answer"] = "Test answer"
        
        summary = get_state_summary(state)
        
        # Check summary contains expected fields
        assert "conversation_id" in summary
        assert "session_id" in summary
        assert "current_step" in summary
        assert summary["current_step"] == WorkflowSteps.REASONING
        assert summary["iteration_count"] == 3
        assert summary["error_count"] == 1
        assert summary["tool_calls_count"] == 2
        assert summary["has_final_answer"] is True


class TestStateErrorHandling:
    """Test error handling in state operations."""
    
    def test_serialize_invalid_state(self):
        """Test serialization with invalid state."""
        # Create state with non-serializable content
        state = get_empty_math_agent_state()
        state["mathematical_context"] = {"function": lambda x: x}  # Non-serializable
        
        # Should handle gracefully
        try:
            serialize_state(state)
        except StateError:
            pass  # Expected
    
    def test_deserialize_invalid_json(self):
        """Test deserialization with invalid JSON."""
        invalid_json = '{"invalid": json}'
        
        with pytest.raises(StateError):
            deserialize_state(invalid_json)
    
    def test_deserialize_missing_required_fields(self):
        """Test deserialization with missing required fields."""
        incomplete_json = '{"session_id": "test"}'  # Missing required fields
        
        with pytest.raises(StateError):
            deserialize_state(incomplete_json)


class TestStateIntegration:
    """Integration tests for state management."""
    
    def test_state_workflow_progression(self):
        """Test state through a typical workflow progression."""
        # Create initial state
        state = create_initial_state("integration-test")
        assert state["current_step"] == WorkflowSteps.START
        
        # Progress to reasoning
        state = update_state_safely(state, {
            "current_step": WorkflowSteps.REASONING,
            "current_reasoning": "Analyzing the problem",
            "iteration_count": 1
        })
        
        # Progress to tool selection
        state = update_state_safely(state, {
            "current_step": WorkflowSteps.TOOL_SELECTION,
            "available_tools": ["integral_tool", "plot_tool"],
            "selected_tools": ["integral_tool"]
        })
        
        # Progress to tool execution
        state = update_state_safely(state, {
            "current_step": WorkflowSteps.TOOL_EXECUTION,
            "tool_calls": [{"tool": "integral_tool", "args": {"function": "x^2"}}]
        })
        
        # Complete workflow
        state = update_state_safely(state, {
            "current_step": WorkflowSteps.END,
            "workflow_status": WorkflowStatus.COMPLETED,
            "final_answer": "Integration result: x^3/3 + C"
        })
        
        # Verify final state
        assert state["workflow_status"] == WorkflowStatus.COMPLETED
        assert state["current_step"] == WorkflowSteps.END
        assert state["final_answer"] is not None
        assert len(state["tool_calls"]) == 1
        
        # Should still be valid
        assert validate_state(state) is True
    
    def test_state_error_recovery(self):
        """Test state error handling and recovery."""
        state = create_initial_state("error-test")
        
        # Simulate error
        state = update_state_safely(state, {
            "error_count": 1,
            "last_error": "Tool execution failed",
            "error_history": [{"error": "Tool execution failed", "timestamp": datetime.now().isoformat()}]
        })
        
        # Attempt recovery
        state = update_state_safely(state, {
            "recovery_attempts": 1,
            "current_step": WorkflowSteps.TOOL_SELECTION,  # Retry tool selection
            "last_error": None  # Clear error
        })
        
        assert state["error_count"] == 1  # Error count preserved
        assert state["recovery_attempts"] == 1
        assert state["last_error"] is None
        assert len(state["error_history"]) == 1
        
        # Should still be valid
        assert validate_state(state) is True
