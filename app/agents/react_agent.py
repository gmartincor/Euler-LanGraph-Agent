"""ReAct Mathematical Agent Core.

This module implements the core ReAct (Reasoning and Acting) agent for 
mathematical problem solving, integrating with existing infrastructure
and following professional design patterns.

Key features:
- Integration with existing ToolRegistry and BigTool
- Reuse of configuration and logging systems  
- Professional error handling and recovery
- Mathematical problem specialization
- LangGraph integration for workflow management
"""

from typing import Any, Dict, List, Optional, Union, Callable
from uuid import uuid4
from datetime import datetime
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from ..core.config import Settings, get_settings
from ..core.logging import get_logger, log_function_call, correlation_context
from ..core.exceptions import AgentError, ToolError, ValidationError
from ..tools.registry import ToolRegistry
from ..core.bigtool_setup import BigToolManager, create_bigtool_manager
from ..models.agent_state import AgentMemory
from .state import MathAgentState
from .state_utils import create_initial_state, validate_state, update_state_safely
from .chains import ChainFactory, create_chain_factory, create_all_chains
from .prompts import get_prompt_template

logger = get_logger(__name__)


class ReactMathematicalAgent:
    """
    Mathematical ReAct agent with LangGraph integration.
    
    This agent specializes in mathematical problem solving using the ReAct
    (Reasoning and Acting) methodology. It integrates with existing infrastructure
    to provide a complete mathematical reasoning system.
    
    Key capabilities:
    - Step-by-step mathematical reasoning
    - Intelligent tool selection and usage
    - Result validation and reflection
    - Error recovery and alternative approaches
    - Integration with existing mathematical tools
    """
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        tool_registry: Optional[ToolRegistry] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize the ReAct Mathematical Agent.
        
        Args:
            settings: Application settings (uses default if None)
            tool_registry: Tool registry (uses default if None)
            checkpointer: Checkpoint saver for persistence
            session_id: Optional session identifier
        """
        # Robust error handling during construction (fail-safe pattern)  
        try:
            self.settings = settings or get_settings()
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}, using minimal defaults")
            # Create minimal settings object for fail-safe operation
            from ..core.config import Settings
            self.settings = Settings()
            
        self.tool_registry = tool_registry or self._get_default_tool_registry()
        self.checkpointer = checkpointer
        self.session_id = session_id or str(uuid4())
        
        # Core components (initialized lazily)
        self._llm: Optional[ChatGoogleGenerativeAI] = None
        self._bigtool_manager: Optional[BigToolManager] = None
        self._chain_factory: Optional[ChainFactory] = None
        self._chains: Dict[str, Any] = {}
        self._graph: Optional[StateGraph] = None
        self._compiled_agent = None
        
        # State tracking
        self._is_initialized = False
        self._current_state: Optional[MathAgentState] = None
        
        logger.info(f"ReactMathematicalAgent created with session_id: {self.session_id}")
    
    def _get_default_tool_registry(self) -> ToolRegistry:
        """Get default tool registry with error handling."""
        try:
            from ..tools.initialization import get_tool_registry
            return get_tool_registry()
        except Exception as e:
            logger.error(f"Failed to get default tool registry: {e}")
            from ..tools.registry import ToolRegistry
            return ToolRegistry()
    
    @log_function_call(logger)
    async def initialize(self) -> None:
        """
        Initialize the agent with essential components only (YAGNI principle).
        
        This method sets up only what's actually needed:
        - LLM configuration
        - BigTool manager for semantic tool search  
        - Chain factory for reasoning processes
        - LangGraph workflow orchestration
        
        Raises:
            AgentError: If initialization fails with detailed error context
        """
        if self._is_initialized:
            logger.info("Agent already initialized, skipping")
            return
            
        try:
            with correlation_context(agent_id=self.session_id):
                logger.info("Initializing ReactMathematicalAgent with professional patterns")
                
                # Step 1: Core LLM (essential)
                self._llm = self._create_llm()
                logger.debug("✅ LLM initialized")
                
                # Step 2: BigTool manager (optional but valuable)
                try:
                    self._bigtool_manager = await create_bigtool_manager(
                        self.tool_registry,
                        self.settings
                    )
                    logger.debug("✅ BigTool manager initialized")
                except Exception as e:
                    logger.warning(f"BigTool manager initialization failed: {e}")
                    self._bigtool_manager = None  # Graceful degradation
                
                # Step 3: Chain factory (essential for reasoning)
                self._chain_factory = create_chain_factory(
                    self.settings,
                    self.tool_registry,
                    self._llm
                )
                self._chains = create_all_chains(self._chain_factory)
                logger.debug("✅ Reasoning chains initialized")
                
                # Step 4: Workflow orchestration (essential)
                self._graph = self._create_langgraph_workflow()
                logger.debug("✅ LangGraph workflow created")
                
                # Step 5: Compile agent (essential)
                self._compiled_agent = self._graph.compile(
                    checkpointer=self.checkpointer
                )
                logger.debug("✅ Agent compiled successfully")
                
                self._is_initialized = True
                logger.info("ReactMathematicalAgent initialized successfully")
                
        except Exception as e:
            # Reset state on failure (fail-safe behavior)
            self._is_initialized = False
            self._llm = None
            self._bigtool_manager = None  
            self._chain_factory = None
            self._chains = {}
            self._graph = None
            self._compiled_agent = None
            
            error_msg = f"Failed to initialize ReactMathematicalAgent: {str(e)}"
            logger.error(error_msg, exc_info=True) 
            raise AgentError(error_msg) from e
    
    def _create_llm(self) -> ChatGoogleGenerativeAI:
        """
        Create and configure the Gemini LLM.
        
        Returns:
            ChatGoogleGenerativeAI: Configured LLM instance
        """
        gemini_config = self.settings.gemini_config
        
        llm = ChatGoogleGenerativeAI(
            model=gemini_config["model_name"],
            api_key=gemini_config["api_key"],
            temperature=gemini_config["temperature"],
            max_output_tokens=gemini_config["max_tokens"],
            convert_system_message_to_human=True,  # Required for Gemini
        )
        
        logger.info(f"LLM configured: {gemini_config['model_name']}")
        return llm
    
    def _create_langgraph_workflow(self) -> StateGraph:
        """
        Create LangGraph workflow using professional orchestration.
        
        This method uses the extracted MathematicalAgentGraph orchestrator
        following DRY principles and clean architecture patterns.
        
        Returns:
            StateGraph: Complete ReAct workflow via extracted orchestration
            
        Raises:
            AgentError: If workflow creation fails
        """
        try:
            from .graph import create_mathematical_agent_graph
            
            # Use extracted orchestration (DRY principle - single source of truth)
            graph_orchestrator = create_mathematical_agent_graph(self)
            workflow = graph_orchestrator.build_graph()
            
            logger.info("LangGraph workflow created via extracted orchestration")
            return workflow
            
        except ImportError as e:
            error_msg = f"Graph orchestration module not available: {e}"
            logger.error(error_msg)
            raise AgentError(f"Missing workflow orchestration dependencies: {e}") from e
            
        except Exception as e:
            error_msg = f"Failed to create workflow via orchestrator: {e}"
            logger.error(error_msg, exc_info=True)
            raise AgentError(f"Workflow creation failed: {e}") from e
    
    # === Conditional Edge Functions (Delegated to Extracted Module) ===
    
    def _should_use_tools(self, state: MathAgentState) -> str:
        """Delegate to extracted conditions module (DRY principle)."""
        from .conditions import should_use_tools
        return should_use_tools(state, self)
    
    def _should_continue_reasoning(self, state: MathAgentState) -> str:
        """Delegate to extracted conditions module (DRY principle)."""
        from .conditions import should_continue_reasoning  
        return should_continue_reasoning(state, self)
    
    def _should_finalize(self, state: MathAgentState) -> str:
        """Delegate to extracted conditions module (DRY principle)."""
        from .conditions import should_finalize
        return should_finalize(state, self)
    
    def _should_retry(self, state: MathAgentState) -> str:
        """Delegate to extracted conditions module (DRY principle)."""
        from .conditions import should_retry
        return should_retry(state, self)

    # === Node Implementations (Complete Delegation Pattern) ===
    
    async def _delegate_to_extracted_node(
        self, 
        state: MathAgentState, 
        node_name: str
    ) -> Dict[str, Any]:
        """
        Professional delegation pattern for node execution.
        
        This method implements complete delegation to extracted nodes,
        eliminating code duplication and following DRY principles.
        
        Args:
            state: Current agent state
            node_name: Name of the extracted node function
            
        Returns:
            Dict[str, Any]: State updates from the extracted node
            
        Raises:
            AgentError: If node execution fails
        """
        try:
            from .nodes import (
                analyze_problem_node, reasoning_node, tool_action_node,
                validation_node, final_response_node, error_recovery_node
            )
            
            # Get the appropriate node function (Strategy Pattern)
            node_function = {
                'analyze_problem': analyze_problem_node,
                'reasoning': reasoning_node,
                'tool_action': tool_action_node,
                'validation': validation_node,
                'final_response': final_response_node,
                'error_recovery': error_recovery_node
            }.get(node_name)
            
            if node_function:
                return await node_function(state, self)
            else:
                raise AgentError(f"Unknown node function: {node_name}")
                
        except ImportError as e:
            error_msg = f"Extracted nodes not available for {node_name}: {e}"
            logger.error(error_msg)
            raise AgentError(f"Missing node dependencies: {e}") from e
        except Exception as e:
            error_msg = f"Node execution failed for {node_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise AgentError(error_msg) from e
    
    async def _analyze_problem_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Analyze problem (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'analyze_problem')
    
    async def _reasoning_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Perform reasoning (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'reasoning')
    
    async def _tool_action_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Execute tools (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'tool_action')
    
    async def _validation_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Validate results (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'validation')
    
    async def _final_response_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Generate response (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'final_response')
    
    async def _error_recovery_node(self, state: MathAgentState) -> Dict[str, Any]:
        """Handle errors (DRY - complete delegation to extracted node)."""
        return await self._delegate_to_extracted_node(state, 'error_recovery')
    
    # === Utility Methods (Core Logic) ===
    
    def _extract_problem_from_messages(self, messages: List[BaseMessage]) -> str:
        """
        Extract the mathematical problem from messages using KISS principle.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            str: Extracted problem text or empty string
        """
        if not messages:
            return ""
            
        # Get the most recent human message (KISS - simple and direct)
        for msg in reversed(messages):
            if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                return msg.content.strip()
        
        return ""
    
    def _extract_tool_parameters(self, state: MathAgentState, tool_name: str) -> Dict[str, Any]:
        """
        Extract parameters for tool execution (simplified for KISS).
        
        In production, this would use LLM-based parameter extraction,
        but for now we provide sensible defaults based on problem context.
        """
        # YAGNI - Only implement what's actually needed
        problem_type = state.get("problem_type", "calculus")
        
        # Simple heuristics based on problem type (KISS approach)
        if "integral" in state.get("current_problem", "").lower():
            return {
                "expression": "x^2",  # Default function
                "variable": "x",
                "lower_bound": 0,
                "upper_bound": 1
            }
        elif "derivative" in state.get("current_problem", "").lower():
            return {
                "expression": "x^2", 
                "variable": "x"
            }
        else:
            # Generic mathematical expression
            return {
                "expression": "x^2",
                "variable": "x"
            }
    
    def _extract_final_answer(self, state: MathAgentState) -> str:
        """
        Extract final answer from state using KISS principle.
        
        Args:
            state: Current agent state
            
        Returns:
            str: Final answer or appropriate fallback message
        """
        # Check tool results first (most reliable source)
        tool_results = state.get("tool_results", [])
        if tool_results:
            # Get the last successful result with robust None handling
            for result in reversed(tool_results):
                # Skip None values to prevent AttributeError
                if result is not None and result.get("success") and result.get("result"):
                    return str(result["result"])
        
        # Fallback to reasoning steps if no tool results
        reasoning_steps = state.get("reasoning_steps", [])
        if reasoning_steps:
            # Look for answer patterns in reasoning (simplified)
            last_step = reasoning_steps[-1]
            if "answer" in last_step.lower() or "result" in last_step.lower():
                return last_step
        
        # Final fallback
        return "No final answer available - problem may need further analysis"
    
    # === Public Interface ===
    
    @log_function_call(logger)
    async def solve_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve a mathematical problem using ReAct reasoning.
        
        Args:
            problem: Mathematical problem to solve
            context: Optional additional context
            user_id: Optional user identifier
            
        Returns:
            Dict[str, Any]: Solution result
            
        Raises:
            AgentError: If agent is not initialized or solving fails
        """
        if not self._is_initialized:
            raise AgentError("Agent must be initialized before solving problems")
        
        try:
            # Create initial state
            initial_state = create_initial_state(
                session_id=self.session_id,
                user_id=user_id,
                max_iterations=self.settings.agent_max_iterations
            )
            
            # Add problem message
            initial_state["messages"] = [HumanMessage(content=problem)]
            initial_state["mathematical_context"] = context or {}
            
            # Run the agent
            config = {"configurable": {"thread_id": self.session_id}}
            result = await self._compiled_agent.ainvoke(initial_state, config=config)
            
            return {
                "success": True,
                "final_answer": result.get("final_answer", "No answer"),
                "confidence_score": result.get("confidence_score", 0.0),
                "tools_used": [call["tool_name"] for call in result.get("tool_calls", [])],
                "reasoning_steps": result.get("reasoning_steps", []),
                "session_id": self.session_id
            }
            
        except Exception as e:
            error_msg = f"Problem solving failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise AgentError(error_msg) from e
    
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for current session."""
        if not self._compiled_agent or not self.checkpointer:
            return []
        
        try:
            config = {"configurable": {"thread_id": self.session_id}}
            history = []
            
            # Get checkpoint history (simplified)
            # In real implementation, would iterate through checkpoints
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._is_initialized
    
    @property
    def available_tools(self) -> List[str]:
        """Get list of available tools (KISS - simple property access)."""
        try:
            return self.tool_registry.list_tools()
        except Exception as e:
            logger.warning(f"Failed to get available tools: {e}")
            return []
    
    @property 
    def session_info(self) -> Dict[str, Any]:
        """Get essential session information (YAGNI - only what's needed)."""
        return {
            "session_id": self.session_id,
            "initialized": self._is_initialized,
            "available_tools": len(self.available_tools),
            "bigtool_enabled": self._bigtool_manager is not None,
            "chains_loaded": len(self._chains) if self._chains else 0
        }


# === Factory Functions (DRY and Professional) ===

@log_function_call(logger)
async def create_react_agent(
    settings: Optional[Settings] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    session_id: Optional[str] = None
) -> ReactMathematicalAgent:
    """
    Factory function to create and initialize a ReAct Mathematical Agent.
    
    This factory follows professional patterns:
    - Single Responsibility: Only creates and initializes agents
    - Dependency Injection: All dependencies can be provided
    - Fail Fast: Throws clear errors if initialization fails
    - DRY: Reuses existing initialization logic
    
    Args:
        settings: Application settings (uses defaults if None)
        tool_registry: Tool registry (uses defaults if None)  
        checkpointer: Checkpoint saver for persistence (optional)
        session_id: Session identifier (auto-generated if None)
        
    Returns:
        ReactMathematicalAgent: Fully initialized and ready-to-use agent
        
    Raises:
        AgentError: If agent creation or initialization fails
    """
    try:
        # Create agent instance (KISS - simple instantiation)
        agent = ReactMathematicalAgent(
            settings=settings,
            tool_registry=tool_registry,
            checkpointer=checkpointer,
            session_id=session_id
        )
        
        # Initialize agent (DRY - reuse existing initialization logic)
        await agent.initialize()
        
        logger.info(f"ReactMathematicalAgent created and initialized: {agent.session_id}")
        return agent
        
    except Exception as e:
        error_msg = f"Failed to create ReactMathematicalAgent: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise AgentError(error_msg) from e
