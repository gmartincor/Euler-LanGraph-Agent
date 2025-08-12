from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict

# Removed circular import - use direct function if needed
__all__ = ['AgentMemory', 'AgentConversation']


class AgentMemory(BaseModel):
    
    model_config = ConfigDict()
    
    short_term: Dict[str, Any] = Field(default_factory=dict, description="Short-term memory")
    long_term: Dict[str, Any] = Field(default_factory=dict, description="Long-term memory")
    context_window: List[str] = Field(default_factory=list, description="Recent context messages")
    max_context_size: int = Field(default=50, description="Maximum context window size")
    
    def add_to_context(self, message: str) -> None:
        self.context_window.append(message)
        
        # Maintain context window size
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def clear_context(self) -> None:
        self.context_window.clear()
    
    def update_short_term(self, key: str, value: Any) -> None:
        self.short_term[key] = value
    
    def update_long_term(self, key: str, value: Any) -> None:
        self.long_term[key] = value
    
    def get_memory_summary(self) -> Dict[str, Any]:
        return {
            "short_term_keys": list(self.short_term.keys()),
            "long_term_keys": list(self.long_term.keys()),
            "context_size": len(self.context_window),
            "max_context_size": self.max_context_size,
        }


class ToolState(BaseModel):
    
    available_tools: List[str] = Field(default_factory=list, description="List of available tool names")
    tool_usage_count: Dict[str, int] = Field(default_factory=dict, description="Usage count per tool")
    last_used_tool: Optional[str] = Field(None, description="Name of the last used tool")
    tool_preferences: Dict[str, float] = Field(default_factory=dict, description="Tool preference scores")
    
    def record_tool_usage(self, tool_name: str) -> None:
        self.tool_usage_count[tool_name] = self.tool_usage_count.get(tool_name, 0) + 1
        self.last_used_tool = tool_name
    
    def get_tool_stats(self) -> Dict[str, Any]:
        total_usage = sum(self.tool_usage_count.values())
        return {
            "total_tool_calls": total_usage,
            "unique_tools_used": len(self.tool_usage_count),
            "most_used_tool": max(self.tool_usage_count, key=self.tool_usage_count.get) if self.tool_usage_count else None,
            "tool_usage_distribution": self.tool_usage_count,
        }


class ConversationContext(BaseModel):
    
    current_topic: Optional[str] = Field(None, description="Current conversation topic")
    math_expressions: List[str] = Field(default_factory=list, description="Mathematical expressions discussed")
    solved_problems: List[Dict[str, Any]] = Field(default_factory=list, description="Previously solved problems")
    user_preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    session_metadata: Dict[str, Any] = Field(default_factory=dict, description="Session-specific metadata")
    
    def add_math_expression(self, expression: str) -> None:
        if expression not in self.math_expressions:
            self.math_expressions.append(expression)
    
    def add_solved_problem(
        self,
        problem: str,
        solution: str,
        method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        problem_data = {
            "problem": problem,
            "solution": solution,
            "method": method,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.solved_problems.append(problem_data)
    
    def get_related_problems(self, current_problem: str) -> List[Dict[str, Any]]:
        # Simple keyword-based matching for now
        keywords = current_problem.lower().split()
        related = []
        
        for problem in self.solved_problems:
            problem_text = problem["problem"].lower()
            if any(keyword in problem_text for keyword in keywords):
                related.append(problem)
        
        return related


class AgentState(BaseModel):
    
    id: UUID = Field(default_factory=uuid4, description="Unique state identifier")
    session_id: str = Field(..., description="Session identifier")
    memory: AgentMemory = Field(default_factory=AgentMemory, description="Agent memory")
    tool_state: ToolState = Field(default_factory=ToolState, description="Tool state")
    conversation_context: ConversationContext = Field(
        default_factory=ConversationContext, description="Conversation context"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional state metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    version: int = Field(default=1, description="State version for conflict resolution")
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()
    
    def update_state(self) -> None:
        self.updated_at = datetime.utcnow()
        self.version += 1
    
    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "memory_summary": self.memory.get_memory_summary(),
            "tool_stats": self.tool_state.get_tool_stats(),
            "conversation_topics": [self.conversation_context.current_topic] if self.conversation_context.current_topic else [],
            "problems_solved": len(self.conversation_context.solved_problems),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    def reset_session(self) -> None:
        self.memory = AgentMemory()
        self.tool_state = ToolState()
        self.conversation_context = ConversationContext()
        self.update_state()
    
    model_config = ConfigDict(json_encoders={
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        })


class AgentStateCreate(BaseModel):
    
    session_id: str = Field(..., description="Session identifier")
    memory: Optional[AgentMemory] = Field(None, description="Initial memory state")
    tool_state: Optional[ToolState] = Field(None, description="Initial tool state")
    conversation_context: Optional[ConversationContext] = Field(None, description="Initial conversation context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="State metadata")
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()


class AgentStateUpdate(BaseModel):
    
    memory: Optional[AgentMemory] = Field(None, description="Updated memory state")
    tool_state: Optional[ToolState] = Field(None, description="Updated tool state")
    conversation_context: Optional[ConversationContext] = Field(None, description="Updated conversation context")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated state metadata")
    
    model_config = ConfigDict(json_encoders={
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        })
