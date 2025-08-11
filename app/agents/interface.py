from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime
from uuid import uuid4

from ..core.logging import get_logger, log_function_call
from ..core.config import get_settings, Settings
from ..core.exceptions import AgentError, ValidationError
from ..tools.registry import ToolRegistry
from .graph import MathematicalAgentGraph
from .state_utils import create_initial_state, format_agent_response
from .checkpointer import create_checkpointer

logger = get_logger(__name__)


class MathematicalAgent:
    """Mathematical Agent with unified LangGraph workflow for problem solving."""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        session_id: Optional[str] = None,
        enable_persistence: bool = True
    ):
        """Initialize mathematical agent with settings and persistence options."""
        self.settings = settings or get_settings()
        self.session_id = session_id or str(uuid4())
        self.enable_persistence = enable_persistence
        
        self.tool_registry = ToolRegistry()
        # Use memory checkpointer to avoid async initialization issues in constructor
        try:
            if enable_persistence:
                from .checkpointer import create_memory_checkpointer
                self.checkpointer = create_memory_checkpointer()
            else:
                self.checkpointer = None
        except Exception as e:
            logger.warning(f"Failed to initialize checkpointer: {e}")
            self.checkpointer = None
        
        self.workflow_graph = MathematicalAgentGraph(
            settings=self.settings,
            tool_registry=self.tool_registry,
            checkpointer=self.checkpointer
        )
        
        self._compiled_workflow = None
        
        logger.info(f"Mathematical agent initialized: session={self.session_id}")
    
    @property
    def compiled_workflow(self):
        """Get or create compiled workflow."""
        if self._compiled_workflow is None:
            self._compiled_workflow = self.workflow_graph.compile_graph()
        return self._compiled_workflow
    
    @log_function_call(logger)
    async def solve(
        self, 
        problem: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Solve mathematical problem using complete reasoning workflow."""
        try:
            if not problem or not problem.strip():
                raise ValidationError("Problem cannot be empty")
            
            logger.info(f"Solving problem: {problem[:50]}...")
            
            initial_state = create_initial_state(
                problem=problem.strip(),
                session_id=self.session_id,
                context=context or [],
                **kwargs
            )
            
            start_time = datetime.now()
            raw_result = await self.compiled_workflow.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": self.session_id}}
            )
            execution_time = (datetime.now() - start_time).total_seconds()
            
            formatted_result = format_agent_response(raw_result)
            formatted_result['execution_time'] = execution_time
            formatted_result['session_id'] = self.session_id
            
            logger.info(f"Problem solved successfully in {execution_time:.2f}s")
            return formatted_result
            
        except ValidationError:
            # Re-raise validation errors directly (don't wrap them)
            raise
        except Exception as e:
            error_msg = f"Failed to solve problem: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    @log_function_call(logger)
    async def solve_stream(
        self,
        problem: str,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Solve mathematical problem with real-time streaming updates."""
        try:
            if not problem or not problem.strip():
                raise ValidationError("Problem cannot be empty")
            
            logger.info(f"Starting streaming solve: {problem[:50]}...")
            
            initial_state = create_initial_state(
                problem=problem.strip(),
                session_id=self.session_id,
                context=context or [],
                **kwargs
            )
            
            async for chunk in self.compiled_workflow.astream(
                initial_state,
                config={"configurable": {"thread_id": self.session_id}}
            ):
                formatted_chunk = self._format_stream_chunk(chunk)
                if formatted_chunk:
                    yield formatted_chunk
                    
        except Exception as e:
            error_msg = f"Failed to stream solve problem: {str(e)}"
            logger.error(error_msg)
            yield {"error": error_msg, "status": "failed"}
    
    def _format_stream_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Format streaming chunk for client consumption."""
        try:
            if not chunk:
                return None
                
            current_step = chunk.get('current_step', 'unknown')
            confidence = chunk.get('confidence_score', 0.0)
            
            update = {
                'status': 'in_progress',
                'current_step': current_step,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            if 'reasoning_trace' in chunk and chunk['reasoning_trace']:
                update['latest_reasoning'] = chunk['reasoning_trace'][-1]
            
            if 'tool_results' in chunk and chunk['tool_results']:
                update['tool_count'] = len(chunk['tool_results'])
            
            if 'final_answer' in chunk:
                update['status'] = 'completed'
                update['final_answer'] = chunk['final_answer']
            
            return update
            
        except Exception as e:
            logger.warning(f"Error formatting stream chunk: {e}")
            return None
    
    @log_function_call(logger)
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for current session."""
        try:
            if not self.enable_persistence or not self.checkpointer:
                return []
            
            history = await self.checkpointer.get_conversation_history(self.session_id)
            return history
            
        except Exception as e:
            error_msg = f"Failed to get conversation history: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    @log_function_call(logger)
    async def clear_conversation(self) -> bool:
        """Clear conversation history for current session."""
        try:
            if not self.enable_persistence or not self.checkpointer:
                return True
            
            success = await self.checkpointer.clear_conversation(self.session_id)
            if success:
                logger.info(f"Conversation cleared: session={self.session_id}")
            
            return success
            
        except Exception as e:
            error_msg = f"Failed to clear conversation: {str(e)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from e
    
    def get_available_tools(self) -> List[str]:
        """Get list of available mathematical tools."""
        return self.tool_registry.list_tools()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities."""
        return {
            'session_id': self.session_id,
            'version': '1.0.0',
            'architecture': 'Unified LangGraph',
            'persistence_enabled': self.enable_persistence,
            'available_tools': self.get_available_tools(),
            'capabilities': [
                'integral_calculation',
                'mathematical_reasoning',
                'step_by_step_solutions',
                'visualization',
                'error_recovery'
            ]
        }


def create_mathematical_agent(
    settings: Optional[Settings] = None,
    session_id: Optional[str] = None,
    enable_persistence: bool = True
) -> MathematicalAgent:
    """Create MathematicalAgent instance with proper configuration."""
    return MathematicalAgent(
        settings=settings,
        session_id=session_id,
        enable_persistence=enable_persistence
    )


# Convenience aliases for backward compatibility and ease of use
Agent = MathematicalAgent
create_agent = create_mathematical_agent
