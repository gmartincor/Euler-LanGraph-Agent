"""Conversation models for chat history and message management."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Enumeration for message roles in a conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Represents a tool call made by the agent."""
    
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool that was called")
    input: Dict[str, Any] = Field(..., description="Input parameters for the tool")
    output: Optional[Dict[str, Any]] = Field(None, description="Output from the tool")
    error: Optional[str] = Field(None, description="Error message if tool call failed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute in seconds")
    
    @validator("execution_time")
    def validate_execution_time(cls, v: Optional[float]) -> Optional[float]:
        """Validate execution time is positive."""
        if v is not None and v < 0:
            raise ValueError("Execution time must be positive")
        return v


class Message(BaseModel):
    """Represents a single message in a conversation."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls in this message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    @validator("content")
    def validate_content(cls, v: str) -> str:
        """Validate message content is not empty."""
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }


class Conversation(BaseModel):
    """Represents a complete conversation with metadata."""
    
    id: UUID = Field(default_factory=uuid4, description="Unique conversation identifier")
    session_id: str = Field(..., description="Session identifier for grouping conversations")
    title: Optional[str] = Field(None, description="Optional conversation title")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @validator("session_id")
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()
    
    @property
    def message_count(self) -> int:
        """Get the number of messages in the conversation."""
        return len(self.messages)
    
    @property
    def last_message(self) -> Optional[Message]:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else None
    
    @property
    def duration(self) -> Optional[float]:
        """Get conversation duration in seconds."""
        if not self.messages or len(self.messages) < 2:
            return None
        
        first_message = self.messages[0]
        last_message = self.messages[-1]
        return (last_message.timestamp - first_message.timestamp).total_seconds()
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a new message to the conversation.
        
        Args:
            role: Role of the message sender
            content: Message content
            tool_calls: Optional list of tool calls
            metadata: Optional message metadata
        
        Returns:
            Message: The created message
        """
        message = Message(
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            metadata=metadata or {},
        )
        
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        
        return message
    
    def get_messages_by_role(self, role: MessageRole) -> List[Message]:
        """Get all messages with a specific role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def to_langchain_format(self) -> List[Dict[str, Any]]:
        """Convert conversation to LangChain message format."""
        langchain_messages = []
        
        for message in self.messages:
            langchain_message = {
                "role": message.role.value,
                "content": message.content,
            }
            
            if message.tool_calls:
                langchain_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.input,
                        },
                    }
                    for tc in message.tool_calls
                ]
            
            langchain_messages.append(langchain_message)
        
        return langchain_messages
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation."""
    
    session_id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Optional conversation title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    
    @validator("session_id")
    def validate_session_id(cls, v: str) -> str:
        """Validate session ID format."""
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()


class ConversationUpdate(BaseModel):
    """Schema for updating an existing conversation."""
    
    title: Optional[str] = Field(None, description="Updated conversation title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated conversation metadata")
    
    class Config:
        """Pydantic configuration."""
        # Allow partial updates
        allow_population_by_field_name = True


class ConversationSummary(BaseModel):
    """Summary information about a conversation."""
    
    id: UUID = Field(..., description="Conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    duration: Optional[float] = Field(None, description="Conversation duration in seconds")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }
