"""Professional ReAct Mathematical Agent - Refactored Architecture.

This module implements a professional ReAct agent using the new modular
architecture with core business logic components, eliminating all circular
dependencies and following DRY, KISS, and YAGNI principles.

Key Design Improvements:
- Zero Circular Dependencies: Uses composition over inheritance
- Pure Business Logic: Delegates to specialized core components
- Professional Error Handling: Comprehensive exception management
- High Modularity: Each component has single responsibility
- Testing Friendly: Clean dependency injection
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..core.config import Settings, get_settings
from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ValidationError
from ..core.bigtool_setup import BigToolManager, create_bigtool_manager
from ..tools.registry import ToolRegistry
from ..models.agent_state import AgentMemory
from .state import MathAgentState, WorkflowStatus, WorkflowSteps
from .state_utils import create_initial_state, validate_state, update_state_safely
from .chains import ChainFactory, create_chain_factory
from .core.mathematical_reasoner import MathematicalReasoner
from .core.tool_orchestrator import ToolOrchestrator  
from .core.state_manager import StateManager
from .graph import MathematicalWorkflowGraph, create_mathematical_workflow_graph

logger = get_logger(__name__)


class ReactMathematicalAgent:
    """
    Professional Mathematical ReAct Agent with Modular Architecture.
    
    This agent specializes in mathematical problem solving using the ReAct
    methodology implemented through specialized core components:
    
    - MathematicalReasoner: Pure mathematical reasoning logic
    - ToolOrchestrator: Professional tool selection and execution
    - StateManager: Immutable state management with history
    - WorkflowGraph: LangGraph orchestration without circular dependencies
    
    Architecture Benefits:
    - No Code Duplication: Single source of truth for each responsibility
    - High Testability: Each component independently testable
    - Professional Quality: Production-ready error handling
    - Zero Circular Dependencies: Clean composition patterns
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        settings: Optional[Settings] = None,
        tool_registry: Optional[ToolRegistry] = None,
        bigtool_manager: Optional[BigToolManager] = None
    ):
        """
        Initialize ReactMathematicalAgent with dependency injection.
        
        Args:
            session_id: Optional session identifier
            settings: Application settings
            tool_registry: Tool registry instance
            bigtool_manager: BigTool manager instance
        """
        try:
            self.session_id = session_id or str(uuid4())
            self.settings = settings or get_settings()
            
            # Initialize external dependencies
            self.tool_registry = tool_registry or ToolRegistry()
            self.bigtool_manager = bigtool_manager or create_bigtool_manager()
            
            # Initialize LLM
            self.llm = self._initialize_llm()
            
            # Initialize Chain Factory
            self.chain_factory = create_chain_factory(
                llm=self.llm,
                settings=self.settings
            )
            
            # Initialize Core Components (No Circular Dependencies)
            self.mathematical_reasoner = MathematicalReasoner(
                chain_factory=self.chain_factory,
                settings=self.settings
            )
            
            self.tool_orchestrator = ToolOrchestrator(
                tool_registry=self.tool_registry,
                bigtool_manager=self.bigtool_manager,
                settings=self.settings
            )
            
            self.state_manager = StateManager(
                settings=self.settings
            )
            
            # Initialize Workflow Graph (No Circular Dependencies)
            self.workflow_graph = create_mathematical_workflow_graph()
            
            # Agent metadata
            self.memory = AgentMemory()
            self.created_at = datetime.utcnow()
            
            logger.info(f"ReactMathematicalAgent initialized successfully (session: {self.session_id})")
            
        except Exception as e:
            logger.error(f"Failed to initialize ReactMathematicalAgent: {e}")
            raise AgentError(f"Agent initialization failed: {e}")
    
    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the Google Gemini LLM."""
        try:
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                google_api_key=self.settings.google_api_key,
                temperature=0.1,
                max_tokens=4096
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise AgentError(f"LLM initialization failed: {e}")
    
    @log_function_call(logger)
    async def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solve a mathematical problem using the complete ReAct workflow.
        
        This is the main entry point that orchestrates the entire mathematical
        reasoning process using the modular core components.
        
        Args:
            problem: Mathematical problem to solve
            context: Optional additional context
            
        Returns:
            Dict containing the complete solution and reasoning
        """
        try:
            # Create initial state using StateManager
            initial_state = self.state_manager.create_initial_state(
                problem=problem,
                context=context or {},
                session_id=self.session_id
            )
            
            # Execute the workflow using the graph
            final_state = await self.workflow_graph.execute_workflow(
                initial_state=initial_state,
                config={"configurable": {"thread_id": self.session_id}}
            )
            
            # Extract and format results
            result = self._format_solution_result(final_state)
            
            # Update memory with successful completion
            self.memory.add_conversation(
                user_input=problem,
                agent_response=result.get('final_answer', ''),
                metadata={
                    'solution_steps': len(result.get('reasoning_chain', [])),
                    'tools_used': len(result.get('tool_results', {})),
                    'confidence_score': result.get('confidence_score', 0.0)
                }
            )
            
            logger.info(f"Problem solved successfully (session: {self.session_id})")
            return result
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_answer": None,
                "reasoning_chain": [],
                "confidence_score": 0.0
            }
    
    @log_function_call(logger)
    def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """
        Analyze a mathematical problem to understand its structure and requirements.
        
        Args:
            problem: Mathematical problem to analyze
            
        Returns:
            Dict containing problem analysis
        """
        try:
            return self.mathematical_reasoner.analyze_problem(problem)
        except Exception as e:
            logger.error(f"Error analyzing problem: {e}")
            raise AgentError(f"Problem analysis failed: {e}")
    
    @log_function_call(logger)
    def perform_reasoning(
        self,
        problem: str,
        previous_reasoning: Optional[List[Dict[str, Any]]] = None,
        tool_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform mathematical reasoning on a problem.
        
        Args:
            problem: Mathematical problem
            previous_reasoning: Previous reasoning steps
            tool_results: Results from tool execution
            
        Returns:
            Dict containing reasoning results
        """
        try:
            return self.mathematical_reasoner.perform_reasoning(
                problem=problem,
                previous_reasoning=previous_reasoning or [],
                tool_results=tool_results or {}
            )
        except Exception as e:
            logger.error(f"Error performing reasoning: {e}")
            raise AgentError(f"Mathematical reasoning failed: {e}")
    
    @log_function_call(logger) 
    async def select_and_execute_tools(
        self,
        reasoning_result: Dict[str, Any],
        available_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Select and execute appropriate tools based on reasoning.
        
        Args:
            reasoning_result: Result from mathematical reasoning
            available_tools: Optional list of available tools
            
        Returns:
            Dict containing tool execution results
        """
        try:
            # Tool selection
            selected_tools = await self.tool_orchestrator.select_optimal_tools(
                reasoning_result=reasoning_result,
                available_tools=available_tools
            )
            
            # Tool execution
            execution_results = await self.tool_orchestrator.execute_tools_parallel(
                selected_tools=selected_tools,
                reasoning_context=reasoning_result
            )
            
            return execution_results
            
        except Exception as e:
            logger.error(f"Error in tool selection/execution: {e}")
            raise AgentError(f"Tool operations failed: {e}")
    
    @log_function_call(logger)
    def validate_results(
        self,
        reasoning_result: Dict[str, Any],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate reasoning and tool results for consistency.
        
        Args:
            reasoning_result: Mathematical reasoning result
            tool_results: Tool execution results
            
        Returns:
            Dict containing validation results
        """
        try:
            return self.mathematical_reasoner.validate_results(
                reasoning_result=reasoning_result,
                tool_results=tool_results
            )
        except Exception as e:
            logger.error(f"Error validating results: {e}")
            raise ValidationError(f"Result validation failed: {e}")
    
    def _format_solution_result(self, final_state: MathAgentState) -> Dict[str, Any]:
        """
        Format the final state into a structured solution result.
        
        Args:
            final_state: Final workflow state
            
        Returns:
            Dict containing formatted solution
        """
        return {
            "success": final_state.get('workflow_status') == WorkflowStatus.COMPLETED,
            "final_answer": final_state.get('final_answer'),
            "reasoning_chain": final_state.get('reasoning_chain', []),
            "tool_results": final_state.get('tool_results', {}),
            "confidence_score": final_state.get('confidence_score', 0.0),
            "validation_result": final_state.get('validation_result', {}),
            "metadata": {
                "session_id": self.session_id,
                "iteration_count": final_state.get('iteration_count', 0),
                "error_count": final_state.get('error_count', 0),
                "workflow_status": final_state.get('workflow_status'),
                "current_step": final_state.get('current_step')
            }
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive agent information and status.
        
        Returns:
            Dict containing agent metadata and status
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "settings": {
                "model": "gemini-1.5-pro",
                "temperature": 0.1,
                "max_tokens": 4096
            },
            "components": {
                "mathematical_reasoner": True,
                "tool_orchestrator": True,
                "state_manager": True,
                "workflow_graph": True
            },
            "tools": {
                "registry_tools": len(self.tool_registry.get_all_tools()),
                "bigtool_enabled": self.bigtool_manager is not None
            },
            "memory": {
                "conversation_count": len(self.memory.conversations)
            }
        }
    
    # === Legacy Compatibility Methods (Deprecated) ===
    
    async def initialize(self) -> None:
        """Legacy method - now handled in constructor."""
        logger.warning("initialize() is deprecated - initialization now happens in constructor")
        pass
    
    def is_initialized(self) -> bool:
        """Legacy method - agent is always initialized after construction."""
        return True
    
    @property
    def available_tools(self) -> List[str]:
        """Get list of available tools."""
        try:
            return self.tool_registry.list_tools()
        except Exception as e:
            logger.warning(f"Failed to get available tools: {e}")
            return []
    
    @property 
    def session_info(self) -> Dict[str, Any]:
        """Get essential session information."""
        return {
            "session_id": self.session_id,
            "initialized": True,
            "available_tools": len(self.available_tools),
            "bigtool_enabled": self.bigtool_manager is not None
        }


# === Factory Functions ===

@log_function_call(logger)
def create_react_mathematical_agent(
    session_id: Optional[str] = None,
    settings: Optional[Settings] = None,
    tool_registry: Optional[ToolRegistry] = None,
    bigtool_manager: Optional[BigToolManager] = None
) -> ReactMathematicalAgent:
    """
    Factory function to create ReactMathematicalAgent instances.
    
    Args:
        session_id: Optional session identifier
        settings: Application settings
        tool_registry: Tool registry instance
        bigtool_manager: BigTool manager instance
        
    Returns:
        ReactMathematicalAgent: Configured agent instance
    """
    return ReactMathematicalAgent(
        session_id=session_id,
        settings=settings,
        tool_registry=tool_registry,
        bigtool_manager=bigtool_manager
    )

# Legacy compatibility alias
create_react_agent = create_react_mathematical_agent
