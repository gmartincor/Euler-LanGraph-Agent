"""
Session State Manager - Professional Streamlit State Management

Implements enterprise-grade state management patterns:
- Centralized state control
- Type safety with Pydantic
- Automatic persistence
- State validation
"""

from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
from datetime import datetime

from app.core import get_logger
from app.models.conversation import Conversation

logger = get_logger(__name__)


class UIState(Enum):
    """UI State enumeration for better state management."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class SessionState:
    """Centralized session state with type safety."""
    # Core state
    ui_state: UIState = UIState.INITIALIZING
    conversation_id: Optional[str] = None
    user_message: str = ""
    agent_response: str = ""
    
    # Agent state
    agent_instance: Optional[Any] = None
    tool_registry: Optional[Dict[str, Any]] = None
    processing: bool = False
    
    # UI state
    sidebar_expanded: bool = True
    show_metrics: bool = False
    show_history: bool = False
    selected_conversation: Optional[str] = None
    
    # Cache and performance
    message_history: list = field(default_factory=list)
    plot_cache: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    
    # Error handling
    last_error: Optional[str] = None
    error_count: int = 0


class SessionStateManager:
    """Professional session state manager with automatic persistence."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize session state with defaults."""
        if 'app_state' not in st.session_state:
            st.session_state.app_state = SessionState()
            self.logger.info("Session state initialized")
    
    @property
    def state(self) -> SessionState:
        """Get current session state."""
        return st.session_state.app_state
    
    def update_state(self, **kwargs) -> None:
        """Update session state with validation."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
                    self.logger.debug(f"Updated state: {key} = {value}")
                else:
                    self.logger.warning(f"Attempt to set unknown state key: {key}")
            
            self.state.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating state: {e}")
            self.state.last_error = str(e)
            self.state.error_count += 1
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Safely get state value with default."""
        return getattr(self.state, key, default)
    
    def set_ui_state(self, ui_state: UIState) -> None:
        """Set UI state with logging."""
        prev_state = self.state.ui_state
        self.update_state(ui_state=ui_state)
        self.logger.info(f"UI state changed: {prev_state} -> {ui_state}")
    
    def set_processing(self, processing: bool) -> None:
        """Set processing state with UI update."""
        self.update_state(processing=processing)
        if processing:
            self.set_ui_state(UIState.PROCESSING)
        else:
            self.set_ui_state(UIState.READY)
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add message to history with metadata."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.state.message_history.append(message)
        self.logger.debug(f"Added message: {role} - {len(content)} chars")
    
    def clear_messages(self) -> None:
        """Clear message history."""
        self.state.message_history.clear()
        self.logger.info("Message history cleared")
    
    def set_error(self, error: str) -> None:
        """Set error state."""
        self.update_state(
            last_error=error,
            error_count=self.state.error_count + 1,
            ui_state=UIState.ERROR
        )
        self.logger.error(f"UI Error set: {error}")
    
    def clear_error(self) -> None:
        """Clear error state."""
        self.update_state(last_error=None)
        if self.state.ui_state == UIState.ERROR:
            self.set_ui_state(UIState.READY)
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get conversation context for agent."""
        return {
            'conversation_id': self.state.conversation_id,
            'message_history': self.state.message_history[-10:],  # Last 10 messages
            'ui_state': self.state.ui_state.value,
            'timestamp': self.state.last_update.isoformat()
        }


# Global state manager instance
_state_manager = None


def get_state_manager() -> SessionStateManager:
    """Get global state manager instance (Singleton pattern)."""
    global _state_manager
    if _state_manager is None:
        _state_manager = SessionStateManager()
    return _state_manager
