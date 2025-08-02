"""Agent state definition for LangGraph ReAct Agent.

This module defines the state structure for the mathematical ReAct agent,
following LangGraph patterns and integrating with existing infrastructure.

The state is designed to be:
- Serializable for checkpointing
- Compatible with existing models
- Extensible for future features
- Thread-safe for concurrent operations
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict
from uuid import UUID, uuid4
from datetime import datetime

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..models.agent_state import AgentMemory


class MathAgentState(TypedDict):
    """
    Complete state definition for the mathematical ReAct agent.
    
    This state integrates with LangGraph's message handling and checkpointing
    while reusing existing infrastructure like AgentMemory for consistency.
    
    Following the DRY principle, this state reuses existing models and 
    infrastructure rather than duplicating functionality.
    """
    
    # === LangGraph Standard Fields ===
    # Messages handled by LangGraph's add_messages reducer
    messages: Annotated[List[BaseMessage], add_messages]
    
    # === Identification & Session Management ===
    conversation_id: UUID
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    # === Workflow State Management ===
    current_step: str  # Current node in the ReAct workflow
    iteration_count: int  # Number of reasoning iterations
    max_iterations: int  # Maximum allowed iterations
    workflow_status: str  # "active", "completed", "failed", "paused"
    
    # === ReAct Reasoning Components ===
    reasoning_steps: List[str]  # History of reasoning steps
    current_reasoning: Optional[str]  # Current reasoning process
    thought_process: List[Dict[str, Any]]  # Detailed thought trace
    observations: List[str]  # Observations from tool executions
    
    # === Tool Management ===
    available_tools: List[str]  # Tools available in current context
    selected_tools: List[str]  # Tools selected for current problem
    tool_calls: List[Dict[str, Any]]  # History of tool invocations
    tool_results: List[Dict[str, Any]]  # Results from tool executions
    tool_selection_rationale: Optional[str]  # Why these tools were chosen
    
    # === Mathematical Context ===
    mathematical_context: Dict[str, Any]  # Math-specific context
    current_problem: Optional[str]  # Problem being solved
    problem_type: Optional[str]  # Type of mathematical problem
    problem_complexity: Optional[str]  # Estimated complexity level
    domain_knowledge: Dict[str, Any]  # Domain-specific knowledge
    
    # === Results & Outputs ===
    intermediate_results: List[Any]  # Intermediate calculation results
    final_answer: Optional[str]  # Final answer to the problem
    confidence_score: Optional[float]  # Confidence in the answer
    visualizations: List[Dict[str, Any]]  # Generated visualizations
    solution_steps: List[str]  # Step-by-step solution
    
    # === Error Handling & Recovery ===
    error_count: int  # Number of errors encountered
    last_error: Optional[str]  # Last error message
    recovery_attempts: int  # Number of recovery attempts
    error_history: List[Dict[str, Any]]  # Detailed error history
    
    # === Memory Integration ===
    # Reusing existing AgentMemory for consistency (DRY principle)
    agent_memory: Dict[str, Any]  # Serialized AgentMemory state
    memory_context: Dict[str, Any]  # Additional memory context
    
    # === Performance & Monitoring ===
    execution_time: Optional[float]  # Total execution time
    token_usage: Dict[str, int]  # LLM token usage tracking
    performance_metrics: Dict[str, Any]  # Performance metrics
    
    # === Configuration ===
    agent_config: Dict[str, Any]  # Agent configuration snapshot
    
    # === Loop Detection & Control (Professional Pattern) ===
    iteration_count: int  # Current iteration count for loop detection
    max_iterations: int  # Maximum allowed iterations before forced termination


# Type aliases for better code readability
StateUpdate = Dict[str, Any]
StateSnapshot = Dict[str, Any]


def get_empty_math_agent_state() -> MathAgentState:
    """
    Create an empty MathAgentState with sensible defaults.
    
    This function provides a clean slate state that can be used
    as a template for new conversations or for testing.
    
    Returns:
        MathAgentState: Empty state with default values
    """
    now = datetime.now()
    
    return MathAgentState(
        # LangGraph fields
        messages=[],
        
        # Identification
        conversation_id=uuid4(),
        session_id="",
        user_id=None,
        created_at=now,
        updated_at=now,
        
        # Workflow
        current_step="start",
        iteration_count=0,
        max_iterations=10,
        workflow_status="active",
        
        # Reasoning
        reasoning_steps=[],
        current_reasoning=None,
        thought_process=[],
        observations=[],
        
        # Tools
        available_tools=[],
        selected_tools=[],
        tool_calls=[],
        tool_results=[],
        tool_selection_rationale=None,
        
        # Mathematical context
        mathematical_context={},
        current_problem=None,
        problem_type=None,
        problem_complexity=None,
        domain_knowledge={},
        
        # Results
        intermediate_results=[],
        final_answer=None,
        confidence_score=None,
        visualizations=[],
        solution_steps=[],
        
        # Error handling
        error_count=0,
        last_error=None,
        recovery_attempts=0,
        error_history=[],
        
        # Memory
        agent_memory={},
        memory_context={},
        
        # Performance
        execution_time=None,
        token_usage={},
        performance_metrics={},
        
        # Configuration
        agent_config={}
    )


# Workflow status constants for type safety
class WorkflowStatus:
    """Constants for workflow status values."""
    ACTIVE = "active"
    COMPLETED = "completed" 
    FAILED = "failed"
    PAUSED = "paused"


# Step constants for the ReAct workflow
class WorkflowSteps:
    """Constants for workflow step names."""
    START = "start"
    INITIALIZATION = "initialization"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    ANSWER_GENERATION = "answer_generation"
    ERROR_RECOVERY = "error_recovery"
    COMPLETE = "complete"
    END = "end"
