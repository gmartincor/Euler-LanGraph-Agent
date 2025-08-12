from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ConfigDict


class MessageRole(str, Enum):
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    
    id: str = Field(..., description="Unique identifier for the tool call")
    name: str = Field(..., description="Name of the tool that was called")
    input: Dict[str, Any] = Field(..., description="Input parameters for the tool")
    output: Optional[Dict[str, Any]] = Field(None, description="Output from the tool")
    error: Optional[str] = Field(None, description="Error message if tool call failed")
    execution_time: Optional[float] = Field(None, description="Time taken to execute in seconds")
    
    @field_validator("execution_time")
    @classmethod
    def validate_execution_time(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("Execution time must be positive")
        return v


class Message(BaseModel):
    
    id: UUID = Field(default_factory=uuid4, description="Unique message identifier")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls in this message")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    
    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message content cannot be empty")
        return v.strip()
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }
    )


class Conversation(BaseModel):
    
    id: UUID = Field(default_factory=uuid4, description="Unique conversation identifier")
    session_id: str = Field(..., description="Session identifier for grouping conversations")
    title: Optional[str] = Field(None, description="Optional conversation title")
    messages: List[Message] = Field(default_factory=list, description="List of messages in the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()
    
    @property
    def message_count(self) -> int:
        return len(self.messages)
    
    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None
    
    @property
    def duration(self) -> Optional[float]:
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
        return [msg for msg in self.messages if msg.role == role]
    
    def to_langchain_format(self) -> List[Dict[str, Any]]:
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
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }
    )


class ConversationCreate(BaseModel):
    
    session_id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Optional conversation title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Conversation metadata")
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty")
        return v.strip()


class ConversationUpdate(BaseModel):
    
    title: Optional[str] = Field(None, description="Updated conversation title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated conversation metadata")
    
    model_config = ConfigDict(
        # Allow partial updates
        allow_population_by_field_name=True
    )


class ConversationSummary(BaseModel):
    
    id: UUID = Field(..., description="Conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    title: Optional[str] = Field(None, description="Conversation title")
    message_count: int = Field(..., description="Number of messages")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    duration: Optional[float] = Field(None, description="Conversation duration in seconds")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat() + "Z",
            UUID: str,
        }
    )
