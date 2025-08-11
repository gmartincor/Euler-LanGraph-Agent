from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
from datetime import datetime

from app.core import get_logger
from app.core.base_classes import BaseStateManager
from app.models.conversation import Conversation

logger = get_logger(__name__)


class UIState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    COMPLETE = "complete"


@dataclass
class SessionState:
    session_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    ui_state: UIState = UIState.INITIALIZING
    conversation_id: Optional[str] = None
    user_message: str = ""
    agent_response: str = ""
    agent_instance: Optional[Any] = None
    tool_registry: Optional[Dict[str, Any]] = None
    processing: bool = False
    sidebar_expanded: bool = True
    show_metrics: bool = False
    show_history: bool = False
    selected_conversation: Optional[str] = None
    message_history: list = field(default_factory=list)
    plot_cache: Dict[str, Any] = field(default_factory=dict)
    last_update: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    last_error: Optional[str] = None
    error_count: int = 0
    # Agent configuration preferences
    show_detailed_steps: bool = True
    show_visualizations: bool = True
    visualization_style: str = "plotly"


class SessionStateManager(BaseStateManager):
    def __init__(self):
        super().__init__("session")
        self._ensure_initialization()
    
    def _ensure_initialization(self) -> None:
        """Defensively ensure session state is properly initialized"""
        try:
            if 'app_state' not in st.session_state:
                st.session_state.app_state = SessionState()
                self.logger.info("Session state initialized")
            # Verify the app_state is accessible
            _ = st.session_state.app_state
        except Exception as e:
            self.logger.error(f"Failed to initialize session state: {e}")
            # Force re-initialization
            st.session_state.app_state = SessionState()
            self.logger.info("Session state force re-initialized")
    
    @property
    def state(self) -> SessionState:
        """Get session state with defensive initialization"""
        self._ensure_initialization()
        return st.session_state.app_state
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state (required by BaseStateManager)."""
        return self.state.__dict__
    
    def update_state(self, **kwargs) -> None:
        """Update state (enhanced with BaseStateManager logging)."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                old_value = getattr(self.state, key)
                setattr(self.state, key, value)
                # Use inherited logging from BaseStateManager
                self.log_state_change(key, old_value, value)
            else:
                self.logger.warning(f"Attempting to set unknown state attribute: {key}")
        self.state.last_update = datetime.now()
    
    def set_ui_state(self, ui_state: UIState) -> None:
        prev_state = self.state.ui_state
        self.update_state(ui_state=ui_state)
        self.logger.info(f"UI state changed: {prev_state} -> {ui_state}")
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get specific state value with validation."""
        if not self.validate_state_key(key):
            self.logger.warning(f"Invalid state key: {key}")
            return default
        return getattr(self.state, key, default)
    
    def set_ui_state(self, ui_state: UIState) -> None:
        """Set UI state with logging."""
        prev_state = self.state.ui_state
        self.update_state(ui_state=ui_state)
        self.logger.info(f"UI state changed: {prev_state} -> {ui_state}")
    
    def set_processing(self, processing: bool) -> None:
        """Set processing state and update UI accordingly."""
        self.update_state(processing=processing)
        if processing:
            self.set_ui_state(UIState.PROCESSING)
        else:
            self.set_ui_state(UIState.READY)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add message to history with validation."""
        if not isinstance(message, dict):
            self.logger.error(f"Invalid message format: {type(message)}")
            return
        message.setdefault('timestamp', datetime.now())
        message.setdefault('metadata', {})
        self.state.message_history.append(message)
        self.logger.debug(f"Added message: {message.get('role', 'unknown')} - {len(str(message.get('content', '')))} chars")
    
    def clear_messages(self) -> None:
        """Clear message history."""
        self.state.message_history.clear()
        self.logger.info("Message history cleared")
    
    def set_error(self, error: str) -> None:
        """Set error state with metrics."""
        self.update_state(
            error_message=error,
            last_error=error,
            error_count=self.state.error_count + 1,
            ui_state=UIState.ERROR
        )
        self.logger.error(f"UI Error set: {error}")
    
    def clear_error(self) -> None:
        """Clear error state."""
        self.update_state(error_message=None, last_error=None)
        if self.state.ui_state == UIState.ERROR:
            self.set_ui_state(UIState.READY)
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get conversation context for agent processing."""
        namespaced_key = self.get_namespaced_key("conversation_context")
        context = {
            'conversation_id': self.state.conversation_id,
            'message_history': self.state.message_history[-10:],
            'ui_state': self.state.ui_state.value,
            'timestamp': self.state.last_update.isoformat()
        }
        self.logger.debug(f"Generated context with key: {namespaced_key}")
        return context


_state_manager = None


def get_state_manager() -> SessionStateManager:
    """Get singleton state manager with defensive initialization"""
    global _state_manager
    try:
        if _state_manager is None:
            _state_manager = SessionStateManager()
        return _state_manager
    except Exception as e:
        logger.error(f"Error getting state manager: {e}")
        # Force recreation
        _state_manager = SessionStateManager()
        return _state_manager
