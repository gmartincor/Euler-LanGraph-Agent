from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import streamlit as st
from datetime import datetime

from app.core import get_logger
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


class SessionStateManager:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
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
    
    def update_state(self, **kwargs) -> None:
        """Update session state with error handling and validation"""
        try:
            # Ensure state is initialized before updating
            current_state = self.state
            
            for key, value in kwargs.items():
                if hasattr(current_state, key):
                    setattr(current_state, key, value)
                    self.logger.debug(f"Updated state: {key} = {value}")
                else:
                    self.logger.warning(f"Attempt to set unknown state key: {key}")
            
            current_state.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating state: {e}")
            # Try to handle the error gracefully
            try:
                if hasattr(st.session_state, 'app_state'):
                    self.state.last_error = str(e)
                    self.state.error_count += 1
            except:
                # If even that fails, re-initialize
                self._ensure_initialization()
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        return getattr(self.state, key, default)
    
    def set_ui_state(self, ui_state: UIState) -> None:
        prev_state = self.state.ui_state
        self.update_state(ui_state=ui_state)
        self.logger.info(f"UI state changed: {prev_state} -> {ui_state}")
    
    def set_processing(self, processing: bool) -> None:
        self.update_state(processing=processing)
        if processing:
            self.set_ui_state(UIState.PROCESSING)
        else:
            self.set_ui_state(UIState.READY)
    
    def add_message(self, message: Dict[str, Any]) -> None:
        if not isinstance(message, dict):
            self.logger.error(f"Invalid message format: {type(message)}")
            return
        message.setdefault('timestamp', datetime.now())
        message.setdefault('metadata', {})
        self.state.message_history.append(message)
        self.logger.debug(f"Added message: {message.get('role', 'unknown')} - {len(str(message.get('content', '')))} chars")
    
    def clear_messages(self) -> None:
        self.state.message_history.clear()
        self.logger.info("Message history cleared")
    
    def set_error(self, error: str) -> None:
        self.update_state(
            error_message=error,
            last_error=error,
            error_count=self.state.error_count + 1,
            ui_state=UIState.ERROR
        )
        self.logger.error(f"UI Error set: {error}")
    
    def clear_error(self) -> None:
        self.update_state(error_message=None, last_error=None)
        if self.state.ui_state == UIState.ERROR:
            self.set_ui_state(UIState.READY)
    
    def get_conversation_context(self) -> Dict[str, Any]:
        return {
            'conversation_id': self.state.conversation_id,
            'message_history': self.state.message_history[-10:],
            'ui_state': self.state.ui_state.value,
            'timestamp': self.state.last_update.isoformat()
        }


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
