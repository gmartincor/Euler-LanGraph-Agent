import asyncio
import threading
import time
from typing import Dict, Optional, Any
from uuid import uuid4

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ValidationError
from ..core.base_classes import MetricsCollector
from ..core.bigtool_setup import create_bigtool_manager

logger = get_logger(__name__)


class AgentController:
    """Refactored controller using BigTool directly - simplified and efficient."""
    
    def __init__(self, session_id: str):
        """Initialize agent controller for a specific session."""
        self.session_id = session_id
        self._bigtool_manager = None
        self._lock = threading.Lock()
        self._is_processing = False
        self._is_initialized = False
        
        # Initialize metrics collector
        self.metrics = MetricsCollector(prefix=f"agent.{session_id}")
        
        logger.info(f"Agent controller initialized for session: {session_id}")
        self.metrics.record_metric("controller_initialized", 1)
    
    async def _ensure_initialized(self):
        """Ensure BigTool manager is initialized for current event loop."""
        # Always reinitialize to avoid event loop conflicts
        # This fixes "Task attached to a different loop" errors
        self._bigtool_manager = await create_bigtool_manager()
        self._is_initialized = True
        logger.info("BigTool manager initialized for agent controller")
    
    @property
    def is_processing(self) -> bool:
        """Check if the agent is currently processing a request."""
        return self._is_processing
    
    @log_function_call(logger)
    def process_message(self, message: str, context: Optional[list] = None) -> Dict[str, Any]:
        """Process user message using BigTool directly - simplified approach."""
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")
        
        logger.info(f"Processing message: {message[:50]}...")
        
        # Record metrics
        self.metrics.record_metric("messages_received", 1)
        self.metrics.record_metric("message_length", len(message))
        
        try:
            self._is_processing = True
            start_time = time.time()
            
            # Use proper event loop management instead of asyncio.run()
            # This fixes "Event loop is closed" errors between consecutive calls
            import asyncio
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Create new loop if current one is closed
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._process_with_bigtool(message.strip()))
            except RuntimeError:
                # No event loop in current thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self._process_with_bigtool(message.strip()))
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["session_id"] = self.session_id
            
            logger.info(f"Message processed successfully in {processing_time:.2f}s")
            self.metrics.record_metric("messages_processed_successfully", 1)
            self.metrics.record_metric("processing_time", processing_time)
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to process message: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics.record_metric("message_processing_errors", 1)
            raise AgentError(error_msg) from e
        finally:
            self._is_processing = False
    
    async def _process_with_bigtool(self, message: str) -> Dict[str, Any]:
        """Process message using BigTool manager - simplified and fixed."""
        await self._ensure_initialized()
        
        # Use BigTool to process the query
        result = await self._bigtool_manager.process_query(
            query=message,
            config={"recursion_limit": 15}
        )
        
        # BigTool manager already returns a proper dict format
        if result["success"]:
            # Extract response content directly
            response_content = result["result"]
            tool_calls = result.get("tool_calls", [])
            
            # Convert tool calls to tools_used format for UI compatibility
            tools_used = [
                {
                    "name": tool_call.get("tool_name", "unknown"),
                    "status": "success",
                    "arguments": tool_call.get("arguments", {})
                }
                for tool_call in tool_calls
            ]
            
            return {
                "success": True,
                "response": response_content,
                "tools_used": tools_used,
                "metadata": {
                    "agent_type": "bigtool_direct",
                    "tool_calls": tool_calls
                }
            }
        else:
            # Handle BigTool errors
            error_message = result.get("error", "Unknown BigTool error")
            return {
                "success": False,
                "response": f"❌ I encountered an error: {error_message}",
                "error": error_message,
                "metadata": {"agent_type": "bigtool_direct", "error": True}
            }
    
    @log_function_call(logger)
    def reset_agent(self) -> None:
        """Reset the BigTool manager for this session."""
        with self._lock:
            if self._bigtool_manager:
                self._bigtool_manager = None
                self._is_initialized = False
                logger.info(f"BigTool manager reset for session: {self.session_id}")
                self.metrics.record_metric("agent_resets", 1)
    
    def health_check(self) -> Dict[str, Any]:
        """Get health status of the agent controller."""
        base_health = {
            "session_id": self.session_id,
            "is_processing": self._is_processing,
            "is_initialized": self._is_initialized,
            "metrics": self.metrics.get_all_metrics()
        }
        
        if self._bigtool_manager:
            base_health["bigtool_health"] = self._bigtool_manager.health_check()
        
        return base_health
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary for monitoring."""
        return self.metrics.get_all_metrics()
    
    def cleanup(self) -> None:
        """Cleanup resources used by this controller."""
        try:
            with self._lock:
                if self._bigtool_manager:
                    self._bigtool_manager = None
                    self._is_initialized = False
                
            logger.info(f"Agent controller cleaned up for session: {self.session_id}")
            self.metrics.record_metric("controller_cleanups", 1)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.metrics.record_metric("cleanup_errors", 1)
            self.metrics.record_metric("cleanup_errors", 1)


# === Global Controller Registry ===

_controller_registry: Dict[str, AgentController] = {}
_registry_lock = threading.Lock()


@log_function_call(logger)
def get_agent_controller(session_id: Optional[str] = None) -> AgentController:
    """Get or create agent controller for a session."""
    if session_id is None:
        session_id = str(uuid4())
    
    with _registry_lock:
        if session_id not in _controller_registry:
            _controller_registry[session_id] = AgentController(session_id)
            logger.info(f"Created new agent controller for session: {session_id}")
        
        return _controller_registry[session_id]


@log_function_call(logger)
def cleanup_session(session_id: str) -> None:
    """Clean up resources for a specific session."""
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
