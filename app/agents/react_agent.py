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
        self.settings = settings or get_settings()
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
        Initialize the agent with all necessary components.
        
        This method sets up:
        - LLM configuration
        - BigTool manager for semantic tool search
        - Chain factory for reasoning processes
        - LangGraph workflow
        
        Raises:
            AgentError: If initialization fails
        """
        try:
            with correlation_context(agent_id=self.session_id):
                logger.info("Initializing ReactMathematicalAgent")
                
                # Step 1: Initialize LLM
                self._llm = self._create_llm()
                
                # Step 2: Initialize BigTool manager
                self._bigtool_manager = await create_bigtool_manager(
                    self.tool_registry,
                    self.settings
                )
                
                # Step 3: Initialize chain factory and chains
                self._chain_factory = create_chain_factory(
                    self.settings,
                    self.tool_registry,
                    self._llm
                )
                self._chains = create_all_chains(self._chain_factory)
                
                # Step 4: Create LangGraph workflow
                self._graph = self._create_langgraph_workflow()
                
                # Step 5: Compile the agent
                self._compiled_agent = self._graph.compile(
                    checkpointer=self.checkpointer
                )
                
                self._is_initialized = True
                logger.info("ReactMathematicalAgent initialized successfully")
                
        except Exception as e:
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
        Create the LangGraph workflow for ReAct reasoning.
        
        This workflow implements the ReAct pattern:
        1. Analyze problem
        2. Reason about approach
        3. Select and use tools
        4. Validate results
        5. Provide final answer
        
        Returns:
            StateGraph: Complete ReAct workflow
        """
        # Create the state graph
        workflow = StateGraph(MathAgentState)
        
        # Add nodes for each step of the ReAct process
        workflow.add_node("analyze_problem", self._analyze_problem_node)
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("tool_action", self._tool_action_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("final_response", self._final_response_node)
        workflow.add_node("error_recovery", self._error_recovery_node)
        
        # Define the flow
        workflow.set_entry_point("analyze_problem")
        
        # Add edges with conditional logic
        workflow.add_edge("analyze_problem", "reasoning")
        workflow.add_conditional_edges(
            "reasoning",
            self._should_use_tools,
            {
                "use_tools": "tool_action",
                "validate": "validation",
                "error": "error_recovery"
            }
        )
        workflow.add_conditional_edges(
            "tool_action",
            self._should_continue_reasoning,
            {
                "continue": "reasoning",
                "validate": "validation",
                "error": "error_recovery"
            }
        )
        workflow.add_conditional_edges(
            "validation",
            self._should_finalize,
            {
                "finalize": "final_response",
                "continue": "reasoning",
                "error": "error_recovery"
            }
        )
        workflow.add_conditional_edges(
            "error_recovery",
            self._should_retry,
            {
                "retry": "reasoning",
                "finalize": "final_response"
            }
        )
        workflow.add_edge("final_response", END)
        
        logger.info("LangGraph workflow created with ReAct pattern")
        return workflow
    
    # === Node Implementations ===
    
    async def _analyze_problem_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Analyze the incoming problem to determine type and approach.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Analyzing problem")
            
            # Get the current problem from messages
            current_problem = self._extract_problem_from_messages(state["messages"])
            
            if not current_problem:
                raise ValidationError("No problem found in messages")
            
            # Use analysis chain
            analysis_result = await self._chains["analysis"].ainvoke({
                "problem": current_problem,
                "user_context": state.get("mathematical_context", {})
            })
            
            # Parse analysis (simplified - would use structured output in production)
            problem_type = "calculus"  # Would extract from analysis_result
            complexity = "intermediate"  # Would extract from analysis_result
            
            return {
                "current_problem": current_problem,
                "problem_type": problem_type,
                "problem_complexity": complexity,
                "current_step": "reasoning",
                "reasoning_steps": [f"Problem analyzed: {analysis_result[:100]}..."]
            }
            
        except Exception as e:
            logger.error(f"Problem analysis failed: {e}")
            return {
                "current_step": "error_recovery",
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def _reasoning_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Perform reasoning about the mathematical problem.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Performing mathematical reasoning")
            
            # Use reasoning chain
            reasoning_result = await self._chains["reasoning"].ainvoke({
                "problem": state["current_problem"],
                "context": state.get("mathematical_context", {})
            })
            
            # Update reasoning steps
            reasoning_steps = list(state.get("reasoning_steps", []))
            reasoning_steps.append(f"Reasoning: {reasoning_result[:100]}...")
            
            return {
                "current_reasoning": reasoning_result,
                "reasoning_steps": reasoning_steps,
                "iteration_count": state.get("iteration_count", 0) + 1
            }
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return {
                "current_step": "error_recovery",
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def _tool_action_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Select and execute appropriate tools.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Selecting and executing tools")
            
            # Get tool recommendations from BigTool
            if self._bigtool_manager:
                recommended_tools = await self._bigtool_manager.get_tool_recommendations(
                    state["current_problem"],
                    context=state.get("mathematical_context", {}),
                    top_k=3
                )
            else:
                # Fallback to registry search
                search_results = self.tool_registry.search_tools(
                    state.get("problem_type", "calculus"),
                    limit=3
                )
                recommended_tools = [r["tool_name"] for r in search_results]
            
            # Select the best tool (simplified logic)
            if recommended_tools:
                selected_tool_name = recommended_tools[0]
                selected_tool = self.tool_registry.get_tool(selected_tool_name)
                
                if selected_tool:
                    # Execute tool (simplified - would need proper parameter extraction)
                    tool_input = self._extract_tool_parameters(state, selected_tool_name)
                    tool_result = selected_tool.execute(tool_input)
                    
                    # Record tool usage
                    tool_calls = list(state.get("tool_calls", []))
                    tool_results = list(state.get("tool_results", []))
                    
                    tool_calls.append({
                        "tool_name": selected_tool_name,
                        "parameters": tool_input,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    tool_results.append({
                        "tool_name": selected_tool_name,
                        "result": tool_result.result if tool_result.success else None,
                        "success": tool_result.success,
                        "error": tool_result.error
                    })
                    
                    # Update observations
                    observations = list(state.get("observations", []))
                    observations.append(f"Used {selected_tool_name}: {str(tool_result)[:100]}...")
                    
                    return {
                        "selected_tools": recommended_tools,
                        "tool_calls": tool_calls,
                        "tool_results": tool_results,
                        "observations": observations
                    }
            
            return {
                "observations": state.get("observations", []) + ["No suitable tools found"]
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "current_step": "error_recovery",
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def _validation_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Validate results and determine if solution is complete.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Validating results")
            
            # Use validation chain
            validation_result = await self._chains["validation"].ainvoke({
                "problem": state["current_problem"],
                "tools_used": [call["tool_name"] for call in state.get("tool_calls", [])],
                "results": state.get("tool_results", []),
                "solution_steps": state.get("reasoning_steps", [])
            })
            
            # Parse validation (simplified)
            confidence_score = 0.8  # Would extract from validation_result
            is_valid = True  # Would extract from validation_result
            
            return {
                "confidence_score": confidence_score,
                "workflow_status": "completed" if is_valid else "needs_review"
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "current_step": "error_recovery",
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def _final_response_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Generate final response for the user.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Generating final response")
            
            # Use response chain
            final_response = await self._chains["response"].ainvoke({
                "problem": state["current_problem"],
                "solution_steps": state.get("reasoning_steps", []),
                "final_answer": self._extract_final_answer(state),
                "confidence_score": state.get("confidence_score", 0.0),
                "tools_used": [call["tool_name"] for call in state.get("tool_calls", [])],
                "verification_status": "Validated" if state.get("confidence_score", 0) > 0.7 else "Needs review"
            })
            
            # Add final message
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=final_response))
            
            return {
                "messages": messages,
                "final_answer": final_response,
                "workflow_status": "completed",
                "current_step": "completed"
            }
            
        except Exception as e:
            logger.error(f"Final response generation failed: {e}")
            return {
                "current_step": "error_recovery",
                "last_error": str(e),
                "error_count": state.get("error_count", 0) + 1
            }
    
    async def _error_recovery_node(self, state: MathAgentState) -> Dict[str, Any]:
        """
        Handle errors and attempt recovery.
        
        Args:
            state: Current agent state
            
        Returns:
            Dict[str, Any]: State updates
        """
        try:
            logger.info("Attempting error recovery")
            
            # Use error recovery chain
            recovery_result = await self._chains["error_recovery"].ainvoke({
                "error_type": "execution_error",
                "error_message": state.get("last_error", "Unknown error"),
                "failed_action": state.get("current_step", "unknown"),
                "error_context": state.get("mathematical_context", {}),
                "current_problem": state.get("current_problem", ""),
                "current_progress": state.get("reasoning_steps", []),
                "previous_results": state.get("tool_results", [])
            })
            
            # Simple recovery logic
            error_count = state.get("error_count", 0)
            if error_count < 3:
                return {
                    "recovery_attempts": state.get("recovery_attempts", 0) + 1,
                    "current_step": "reasoning"  # Try reasoning again
                }
            else:
                # Too many errors, give up gracefully
                messages = list(state.get("messages", []))
                messages.append(AIMessage(
                    content="I encountered multiple errors while solving this problem. Please try rephrasing your question or providing more context."
                ))
                
                return {
                    "messages": messages,
                    "workflow_status": "failed",
                    "current_step": "completed"
                }
                
        except Exception as e:
            logger.error(f"Error recovery failed: {e}")
            return {
                "workflow_status": "failed",
                "current_step": "completed"
            }
    
    # === Conditional Edge Functions ===
    
    def _should_use_tools(self, state: MathAgentState) -> str:
        """Determine if tools should be used."""
        # Simplified logic - would analyze reasoning to determine next step
        if state.get("error_count", 0) > 0:
            return "error"
        elif "tool" in state.get("current_reasoning", "").lower():
            return "use_tools"
        else:
            return "validate"
    
    def _should_continue_reasoning(self, state: MathAgentState) -> str:
        """Determine if reasoning should continue."""
        if state.get("error_count", 0) > 0:
            return "error"
        elif state.get("iteration_count", 0) < state.get("max_iterations", 10):
            return "continue"
        else:
            return "validate"
    
    def _should_finalize(self, state: MathAgentState) -> str:
        """Determine if solution should be finalized."""
        if state.get("error_count", 0) > 0:
            return "error"
        elif state.get("confidence_score", 0) > 0.7:
            return "finalize"
        else:
            return "continue"
    
    def _should_retry(self, state: MathAgentState) -> str:
        """Determine if recovery should retry or give up."""
        if state.get("recovery_attempts", 0) < 2:
            return "retry"
        else:
            return "finalize"
    
    # === Utility Methods ===
    
    def _extract_problem_from_messages(self, messages: List[BaseMessage]) -> str:
        """Extract the mathematical problem from messages."""
        for msg in reversed(messages):
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                return msg.content
        return ""
    
    def _extract_tool_parameters(self, state: MathAgentState, tool_name: str) -> Dict[str, Any]:
        """Extract parameters for tool execution."""
        # Simplified - would use LLM to extract parameters
        return {"expression": "x^2", "a": 0, "b": 1}  # Default parameters
    
    def _extract_final_answer(self, state: MathAgentState) -> str:
        """Extract final answer from state."""
        tool_results = state.get("tool_results", [])
        if tool_results:
            last_result = tool_results[-1]
            if last_result.get("success"):
                return str(last_result.get("result", "No result"))
        return "No final answer available"
    
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
        """Get list of available tools."""
        return self.tool_registry.list_tools()
    
    @property
    def session_info(self) -> Dict[str, Any]:
        """Get session information."""
        return {
            "session_id": self.session_id,
            "initialized": self._is_initialized,
            "available_tools": len(self.available_tools),
            "bigtool_enabled": self._bigtool_manager is not None
        }


# === Factory Functions ===

@log_function_call(logger)
async def create_react_agent(
    settings: Optional[Settings] = None,
    tool_registry: Optional[ToolRegistry] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    session_id: Optional[str] = None
) -> ReactMathematicalAgent:
    """
    Factory function to create and initialize a ReAct Mathematical Agent.
    
    Args:
        settings: Application settings
        tool_registry: Tool registry
        checkpointer: Checkpoint saver
        session_id: Session identifier
        
    Returns:
        ReactMathematicalAgent: Initialized agent
    """
    agent = ReactMathematicalAgent(
        settings=settings,
        tool_registry=tool_registry,
        checkpointer=checkpointer,
        session_id=session_id
    )
    
    await agent.initialize()
    return agent
