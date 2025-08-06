import asyncio
import threading
import weakref
from typing import Dict, Optional, Any, Callable
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, Future

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ValidationError
from ..agents.interface import MathematicalAgent, create_mathematical_agent

logger = get_logger(__name__)


class AgentController:
    """
    Unified controller for managing mathematical agent instances.
    
    This controller manages the lifecycle of agent instances, provides thread-safe
    operations, and maintains a registry of active agents per session.
    
    Key Features:
    - Thread-safe agent management
    - Session-based agent isolation
    - Asynchronous operation support
    - Professional error handling
    - Resource cleanup and management
    """
    
    def __init__(self, session_id: str):
        """
        Initialize agent controller for a specific session.
        
        Args:
            session_id: Unique identifier for the session
        """
        self.session_id = session_id
        self._agent: Optional[MathematicalAgent] = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"agent-{session_id}")
        self._is_processing = False
        
        logger.info(f"Agent controller initialized for session: {session_id}")
    
    @property
    def agent(self) -> MathematicalAgent:
        """Get or create the mathematical agent instance."""
        if self._agent is None:
            with self._lock:
                if self._agent is None:  # Double-check locking
                    self._agent = create_mathematical_agent(
                        session_id=self.session_id,
                        enable_persistence=True
                    )
                    logger.info("Mathematical agent instance created")
        return self._agent
    
    @property
    def is_processing(self) -> bool:
        """Check if the agent is currently processing a request."""
        return self._is_processing
    
    @log_function_call(logger)
    def process_message(self, message: str, context: Optional[list] = None) -> Dict[str, Any]:
        """
        Process a user message using the mathematical agent.
        
        Args:
            message: User input message
            context: Optional conversation context
            
        Returns:
            Dict containing the agent's response and metadata
            
        Raises:
            AgentError: If processing fails
            ValidationError: If input is invalid
        """
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")
        
        logger.info(f"Processing message: {message[:50]}...")
        
        try:
            self._is_processing = True
            
            # Create and submit async task
            future = self._executor.submit(
                self._async_process_message, 
                message.strip(), 
                context or []
            )
            
            # Wait for result with timeout
            result = future.result(timeout=300)  # 5 minute timeout
            
            logger.info("Message processed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Failed to process message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise AgentError(error_msg) from e
        finally:
            self._is_processing = False
    
    def _async_process_message(self, message: str, context: list) -> Dict[str, Any]:
        """
        Internal method to handle async processing in thread executor.
        
        Args:
            message: User message to process
            context: Conversation context
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async agent processing
                result = loop.run_until_complete(
                    self.agent.solve(message, context)
                )
                
                # Handle both dict and object responses
                if isinstance(result, dict):
                    # The agent returns final_answer, not response
                    response_content = (
                        result.get("final_answer") or 
                        result.get("answer") or 
                        result.get("response") or 
                        "No response received"
                    )
                    
                    # Get steps from solution_steps or steps
                    reasoning_steps = (
                        result.get("solution_steps") or 
                        result.get("steps") or 
                        result.get("reasoning", [])
                    )
                    
                    return {
                        "success": True,
                        "response": response_content,
                        "reasoning": reasoning_steps,
                        "tools_used": result.get("tools_used", []),
                        "visualizations": result.get("visualizations", []),
                        "metadata": result.get("metadata", {}),
                        "session_id": self.session_id
                    }
                else:
                    # Handle object with attributes
                    return {
                        "success": True,
                        "response": getattr(result, 'response', None) or getattr(result, 'answer', "No response received"),
                        "reasoning": getattr(result, 'reasoning_steps', []) or getattr(result, 'steps', []),
                        "tools_used": getattr(result, 'tools_used', []),
                        "visualizations": getattr(result, 'visualizations', []),
                        "metadata": getattr(result, 'metadata', {}),
                        "session_id": self.session_id
                    }
                
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Async processing failed: {e}")
            raise AgentError(f"Agent processing failed: {str(e)}") from e
    
    @log_function_call(logger)
    def reset_agent(self) -> None:
        """Reset the agent instance for this session."""
        with self._lock:
            if self._agent:
                # Cleanup existing agent if needed
                self._agent = None
                logger.info(f"Agent reset for session: {self.session_id}")
    
    def cleanup(self) -> None:
        """Cleanup resources used by this controller."""
        try:
            with self._lock:
                if self._agent:
                    self._agent = None
                
                # Shutdown executor
                self._executor.shutdown(wait=True)
                
            logger.info(f"Agent controller cleaned up for session: {self.session_id}")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# === Global Controller Registry ===

_controller_registry: Dict[str, AgentController] = {}
_registry_lock = threading.Lock()


@log_function_call(logger)
def get_agent_controller(session_id: Optional[str] = None) -> AgentController:
    """
    Get or create an agent controller for a session.
    
    Args:
        session_id: Session identifier (will generate if None)
        
    Returns:
        AgentController: Controller instance for the session
    """
    if session_id is None:
        session_id = str(uuid4())
    
    with _registry_lock:
        if session_id not in _controller_registry:
            _controller_registry[session_id] = AgentController(session_id)
            logger.info(f"Created new agent controller for session: {session_id}")
        
        return _controller_registry[session_id]


@log_function_call(logger)
def cleanup_session(session_id: str) -> None:
    """
    Clean up resources for a specific session.
    
    Args:
        session_id: Session to cleanup
    """
    with _registry_lock:
        if session_id in _controller_registry:
            controller = _controller_registry.pop(session_id)
            controller.cleanup()
            logger.info(f"Session cleaned up: {session_id}")


def cleanup_all_sessions() -> None:
    """Clean up all active sessions."""
    with _registry_lock:
        sessions = list(_controller_registry.keys())
        for session_id in sessions:
            cleanup_session(session_id)
        logger.info("All sessions cleaned up")


def get_active_sessions() -> list:
    """Get list of active session IDs."""
    with _registry_lock:
        return list(_controller_registry.keys())
