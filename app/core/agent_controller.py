import asyncio
import threading
import time
from typing import Dict, Optional, Any
from uuid import uuid4

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ValidationError
from ..core.base_classes import MetricsCollector
from ..core.tool_engine import execute_tool_query, get_tool_engine_health

logger = get_logger(__name__)


class AgentController:
    """Controller for managing agent sessions and processing messages."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._lock = threading.Lock()
        self._is_processing = False
        
        self.metrics = MetricsCollector(prefix=f"agent.{session_id}")
        
        logger.info(f"Agent controller initialized for session: {session_id}")
        self.metrics.record_metric("controller_initialized", 1)
    
    @property
    def is_processing(self) -> bool:
        return self._is_processing
    
    @log_function_call(logger)
    def process_message(self, message: str, context: Optional[list] = None) -> Dict[str, Any]:
        if not message or not message.strip():
            raise ValidationError("Message cannot be empty")
        logger.info(f"Processing message: {message[:50]}...")
        self.metrics.record_metric("messages_received", 1)
        self.metrics.record_metric("message_length", len(message))
        try:
            self._is_processing = True
            start_time = time.time()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(execute_tool_query(message.strip()))
            finally:
                loop.close()
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
    
    @log_function_call(logger)
    def reset_agent(self) -> None:
        with self._lock:
            logger.info(f"Agent reset for session: {self.session_id}")
            self.metrics.record_metric("agent_resets", 1)
    
    def health_check(self) -> Dict[str, Any]:
        base_health = {
            "session_id": self.session_id,
            "is_processing": self._is_processing,
            "metrics": self.metrics.get_all_metrics()
        }
        base_health["tool_engine"] = get_tool_engine_health()
        return base_health
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        return self.metrics.get_all_metrics()
    
    def cleanup(self) -> None:
        try:
            with self._lock:
                logger.info(f"Agent controller cleaned up for session: {self.session_id}")
                self.metrics.record_metric("controller_cleanups", 1)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            self.metrics.record_metric("cleanup_errors", 1)


# Global Controller Registry

_controller_registry: Dict[str, AgentController] = {}
_registry_lock = threading.Lock()

@log_function_call(logger)
def get_agent_controller(session_id: Optional[str] = None) -> AgentController:
    if session_id is None:
        session_id = str(uuid4())
    with _registry_lock:
        if session_id not in _controller_registry:
            _controller_registry[session_id] = AgentController(session_id)
            logger.info(f"Created new agent controller for session: {session_id}")
        return _controller_registry[session_id]

@log_function_call(logger)
def cleanup_session(session_id: str) -> None:
    with _registry_lock:
        if session_id in _controller_registry:
            controller = _controller_registry.pop(session_id)
            controller.cleanup()
            logger.info(f"Session cleaned up: {session_id}")

def cleanup_all_sessions() -> None:
    with _registry_lock:
        sessions = list(_controller_registry.keys())
        for session_id in sessions:
            cleanup_session(session_id)
        logger.info("All sessions cleaned up")

def get_active_sessions() -> list:
    with _registry_lock:
        return list(_controller_registry.keys())
