"""Professional Chain Factory for Mathematical ReAct Agent.

This module provides a unified factory for creating chains used in the ReAct
reasoning process. Implements both LangChain integration and simple fallbacks
for comprehensive compatibility.

Design Principles Applied:
- Factory Pattern: Centralized chain creation
- Dependency Injection: LLM and tools injected  
- Single Responsibility: Each chain has one purpose
- Graceful Degradation: Works with or without LangChain
- Professional Error Handling: Comprehensive exception management
"""

from typing import Any, Dict, List, Optional, Callable, Union
from functools import lru_cache

# Try LangChain imports with graceful fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
    from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda
    from pydantic import BaseModel, Field
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Fallback classes for testing
    class ChatGoogleGenerativeAI: pass
    class RunnableSequence: pass
    class PromptTemplate: pass
    class RunnableLambda: pass

from ..core.config import Settings
from ..core.logging import get_logger, log_function_call
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


class ChainFactory:
    """
    Professional factory class for creating mathematical reasoning chains.
    
    This factory creates chains with:
    - LangChain integration when available
    - Simple fallback functions for testing
    - Consistent interface regardless of backend
    - Professional error handling
    
    Implements Dependency Injection pattern for testability.
    """
    
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry,
        llm: Optional[ChatGoogleGenerativeAI] = None
    ):
        """
        Initialize chain factory with dependencies.
        
        Args:
            settings: Application settings
            tool_registry: Existing tool registry
            llm: Optional pre-configured LLM (for testing)
        """
        self.settings = settings
        self.tool_registry = tool_registry
        self._llm = llm or self._create_llm()
        
        logger.info(f"ChainFactory initialized (LangChain: {LANGCHAIN_AVAILABLE})")
    
    def _create_llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """
        Create Google Gemini LLM instance with configuration.
        
        Returns:
            Optional[ChatGoogleGenerativeAI]: Configured LLM instance or None
        """
        if not LANGCHAIN_AVAILABLE:
            return None
            
        try:
            gemini_config = self.settings.gemini_config
            
            return ChatGoogleGenerativeAI(
                model=gemini_config["model_name"],
                temperature=gemini_config["temperature"],
                max_tokens=gemini_config["max_tokens"],
                api_key=gemini_config["api_key"],
                google_api_key=gemini_config["api_key"]  # For compatibility
            )
        except Exception as e:
            logger.warning(f"Could not create LLM: {e}")
            return None
    
    @property
    def llm(self) -> Optional[ChatGoogleGenerativeAI]:
        """Get the configured LLM instance."""
        return self._llm
    
    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompts."""
        try:
            # Try to get tools from registry
            if hasattr(self.tool_registry, 'get_all_tools'):
                tools = self.tool_registry.get_all_tools()
            elif hasattr(self.tool_registry, 'get_available_tools'):
                tools = self.tool_registry.get_available_tools()
            else:
                # Fallback for mock registry
                return "integral_tool, plot_tool, analysis_tool"
            
            descriptions = []
            for tool in tools:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    descriptions.append(f"- {tool.name}: {tool.description}")
                else:
                    descriptions.append(f"- {str(tool)}")
            return "\n".join(descriptions) if descriptions else "No tools available"
        except Exception as e:
            logger.debug(f"Could not get tool descriptions: {e}")
            return "integral_tool, plot_tool, analysis_tool"
    
    def create_reasoning_chain(self) -> Union[RunnableSequence, Callable]:
        """
        Create mathematical reasoning chain.
        
        Returns:
            Union[RunnableSequence, Callable]: Chain or function for reasoning
        """
        if LANGCHAIN_AVAILABLE and self._llm:
            return self._create_langchain_reasoning_chain()
        else:
            return self._create_fallback_reasoning_chain()
    
    def _create_langchain_reasoning_chain(self) -> RunnableSequence:
        """Create LangChain-based reasoning chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematical reasoning expert. Analyze the given problem and provide a structured approach.

Your task is to:
1. Understand the mathematical problem
2. Determine the best approach to solve it
3. Identify the required tools and steps
4. Provide a confidence assessment

Available tools: {tool_descriptions}

Return a JSON with: approach, steps, tools_needed, confidence"""),
            ("human", """Problem: {problem}

Context: {context}

Provide a structured reasoning approach.""")
        ])
        
        chain = (
            RunnablePassthrough.assign(
                tool_descriptions=lambda x: self._get_tool_descriptions()
            )
            | prompt
            | self._llm
            | StrOutputParser()
        )
        
        logger.info("LangChain reasoning chain created")
        return chain
    
    def _create_fallback_reasoning_chain(self) -> Callable:
        """Create simple fallback reasoning function."""
        async def perform_reasoning(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Perform mathematical reasoning on the problem."""
            try:
                problem = input_data.get("problem", "")
                analysis = input_data.get("analysis", {})
                context = input_data.get("context", [])
                
                problem_type = analysis.get("type", "general")
                
                # Determine mathematical approach
                if problem_type == "integral":
                    approach = "Apply integration techniques using fundamental theorem of calculus"
                    steps = [
                        "Identify the function to integrate",
                        "Find the antiderivative", 
                        "Apply the limits of integration",
                        "Calculate the final result"
                    ]
                    tools_needed = ["integral_tool"]
                elif problem_type == "derivative":
                    approach = "Apply differentiation rules"  
                    steps = [
                        "Identify the function to differentiate",
                        "Apply appropriate differentiation rules",
                        "Simplify the result"
                    ]
                    tools_needed = ["analysis_tool"]
                elif problem_type == "visualization":
                    approach = "Create mathematical visualization"
                    steps = [
                        "Parse the function or data",
                        "Set up coordinate system", 
                        "Generate the plot",
                        "Add annotations and labels"
                    ]
                    tools_needed = ["plot_tool"]
                else:
                    approach = "Apply general mathematical reasoning"
                    steps = [
                        "Analyze the problem structure",
                        "Identify relevant mathematical concepts",
                        "Apply appropriate methods",
                        "Verify the solution"
                    ]
                    tools_needed = ["analysis_tool"]
                
                return {
                    "approach": approach,
                    "steps": steps,
                    "tools_needed": tools_needed,
                    "confidence": 0.8
                }
                
            except Exception as e:
                logger.error(f"Error in reasoning: {e}")
                return {
                    "approach": "Error in reasoning process",
                    "steps": [],
                    "tools_needed": [],
                    "confidence": 0.0
                }
        
        logger.info("Fallback reasoning chain created")
        return perform_reasoning
    
    def create_analysis_chain(self) -> Callable:
        """
        Create problem analysis chain.
        
        Returns:
            Callable: Problem analysis function
        """
        async def analyze_problem(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Analyze mathematical problem and determine approach."""
            try:
                problem = input_data.get("problem", "")
                
                # Simple rule-based analysis
                problem_lower = problem.lower()
                
                # Determine problem type
                if "integral" in problem_lower or "∫" in problem:
                    problem_type = "integral"
                    complexity = "medium"
                elif "derivative" in problem_lower or "d/dx" in problem_lower:
                    problem_type = "derivative"
                    complexity = "low"
                elif "plot" in problem_lower or "graph" in problem_lower or "visualize" in problem_lower:
                    problem_type = "visualization"
                    complexity = "low"
                else:
                    problem_type = "general"
                    complexity = "unknown"
                
                # Basic variable detection
                variables = []
                for char in 'xyzabc':
                    if char in problem_lower:
                        variables.append(char)
                
                # Basic function detection
                functions = []
                for func in ['sin', 'cos', 'tan', 'log', 'exp', 'sqrt']:
                    if func in problem_lower:
                        functions.append(func)
                
                return {
                    "type": problem_type,
                    "variables": variables,
                    "functions": functions,
                    "complexity": complexity,
                    "domain": "real" if problem_type != "unknown" else "unknown"
                }
                
            except Exception as e:
                logger.error(f"Error in analysis: {e}")
                return {
                    "type": "unknown",
                    "variables": [],
                    "functions": [],
                    "complexity": "unknown",
                    "domain": "unknown"
                }
        
        logger.info("Problem analysis chain created")
        return analyze_problem
    
    def create_validation_chain(self) -> Callable:
        """
        Create solution validation chain.
        
        Returns:
            Callable: Validation function
        """
        async def validate_solution(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate mathematical solution and results."""
            try:
                problem = input_data.get("problem", "")
                solution = input_data.get("solution", {})
                tool_results = input_data.get("tool_results", [])
                
                # Basic validation logic
                has_answer = "answer" in solution
                has_steps = "steps" in solution and len(solution["steps"]) > 0
                has_tool_results = len(tool_results) > 0
                
                # Determine if solution looks valid
                is_valid = has_answer and has_steps
                confidence = 0.9 if is_valid and has_tool_results else 0.7 if is_valid else 0.3
                
                issues = []
                suggestions = []
                
                if not has_answer:
                    issues.append("Missing final answer")
                    suggestions.append("Provide a clear final answer")
                
                if not has_steps:
                    issues.append("Missing solution steps")
                    suggestions.append("Show step-by-step solution process")
                
                if not has_tool_results:
                    issues.append("No tool calculations performed")
                    suggestions.append("Use mathematical tools to verify results")
                
                return {
                    "is_valid": is_valid,
                    "confidence": confidence,
                    "issues": issues,
                    "suggestions": suggestions,
                    "final_answer": solution.get("answer", "No answer provided")
                }
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "issues": [f"Validation error: {str(e)}"],
                    "suggestions": ["Review solution and try again"],
                    "final_answer": "Error in validation"
                }
        
        logger.info("Solution validation chain created")
        return validate_solution
    
    def create_tool_selection_chain(self) -> Callable:
        """
        Create tool selection chain (required by tests).
        
        Returns:
            Callable: Tool selection function
        """
        async def select_tools(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Select appropriate tools based on problem analysis."""
            try:
                problem_type = input_data.get("problem_type", "general")
                
                # Map problem types to tools
                tool_mapping = {
                    "integral": ["integral_tool"],
                    "derivative": ["analysis_tool"],
                    "visualization": ["plot_tool"],
                    "general": ["analysis_tool"]
                }
                
                selected_tools = tool_mapping.get(problem_type, ["analysis_tool"])
                
                return {
                    "selected_tools": selected_tools,
                    "reasoning": f"Selected tools for {problem_type} problem"
                }
                
            except Exception as e:
                logger.error(f"Error in tool selection: {e}")
                return {
                    "selected_tools": ["analysis_tool"],
                    "reasoning": f"Default tool selection due to error: {str(e)}"
                }
        
        logger.info("Tool selection chain created")
        return select_tools
    
    def create_error_recovery_chain(self) -> Callable:
        """
        Create error recovery chain.
        
        Returns:
            Callable: Error recovery function
        """
        async def recover_from_error(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Provide error recovery strategies."""
            try:
                problem = input_data.get("problem", "")
                error = input_data.get("error", "")
                previous_attempts = input_data.get("previous_attempts", [])
                
                # Simple error recovery strategies
                recovery_strategies = {
                    "timeout": "Simplify the problem or break it into smaller parts",
                    "validation": "Check input format and try alternative approach",
                    "calculation": "Verify mathematical expressions and try step-by-step approach",
                    "tool": "Try using different tools or manual calculation"
                }
                
                # Determine recovery strategy based on error type
                error_lower = error.lower()
                if "timeout" in error_lower:
                    recovery_strategy = recovery_strategies["timeout"]
                elif "validation" in error_lower or "invalid" in error_lower:
                    recovery_strategy = recovery_strategies["validation"]
                elif "calculation" in error_lower or "math" in error_lower:
                    recovery_strategy = recovery_strategies["calculation"]
                elif "tool" in error_lower:
                    recovery_strategy = recovery_strategies["tool"]
                else:
                    recovery_strategy = "Review the problem and try a different approach"
                
                # Provide alternative approach
                num_attempts = len(previous_attempts)
                if num_attempts == 0:
                    alternative_approach = "Try breaking the problem into smaller steps"
                elif num_attempts == 1:
                    alternative_approach = "Use a different mathematical method or tool"
                else:
                    alternative_approach = "Simplify the problem or seek manual verification"
                
                return {
                    "recovery_strategy": recovery_strategy,
                    "alternative_approach": alternative_approach,
                    "simplified_problem": f"Simplified version: {problem[:100]}..." if len(problem) > 100 else problem,
                    "next_action": "retry" if num_attempts < 2 else "manual_review"
                }
                
            except Exception as e:
                logger.error(f"Error in error recovery: {e}")
                return {
                    "recovery_strategy": "Manual review required",
                    "alternative_approach": "Contact support or review manually",
                    "simplified_problem": problem[:100] if problem else "Unknown problem",
                    "next_action": "manual_review"
                }
        
        logger.info("Error recovery chain created")
        return recover_from_error
    
    def create_response_chain(self) -> Callable:
        """
        Create response formatting chain.
        
        Returns:
            Callable: Response formatting function
        """
        async def format_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Format final response with solution and explanation."""
            try:
                problem = input_data.get("problem", "")
                solution = input_data.get("solution", {})
                tool_results = input_data.get("tool_results", [])
                validation = input_data.get("validation", {})
                reasoning = input_data.get("reasoning", {})
                
                # Extract solution components
                final_answer = solution.get("answer", "No solution available")
                steps = solution.get("steps", [])
                
                # Calculate overall confidence
                solution_confidence = solution.get("confidence", 0.5)
                validation_confidence = validation.get("confidence", 0.5)
                reasoning_confidence = reasoning.get("confidence", 0.5)
                
                confidence = (solution_confidence + validation_confidence + reasoning_confidence) / 3
                
                return {
                    "answer": final_answer,
                    "steps": steps,
                    "explanation": reasoning.get("approach", "Mathematical reasoning applied"),
                    "confidence": confidence
                }
                
            except Exception as e:
                logger.error(f"Error formatting response: {e}")
                return {
                    "answer": "Error processing solution",
                    "steps": [],
                    "explanation": f"Error occurred: {str(e)}",
                    "confidence": 0.0
                }
        
        logger.info("Response formatting chain created")
        return format_response


# === Factory Functions ===

@log_function_call(logger)
def create_chain_factory(
    settings: Settings,
    tool_registry: ToolRegistry,
    llm: Optional[ChatGoogleGenerativeAI] = None
) -> ChainFactory:
    """
    Factory function to create ChainFactory instance.
    
    Args:
        settings: Application settings
        tool_registry: Existing tool registry
        llm: Optional pre-configured LLM (for testing)
        
    Returns:
        ChainFactory: Initialized chain factory
    """
    return ChainFactory(settings, tool_registry, llm)


def create_all_chains(chain_factory: ChainFactory) -> Dict[str, Callable]:
    """
    Create all standard chains for the ReAct agent.
    
    Args:
        chain_factory: Initialized chain factory
        
    Returns:
        Dict[str, Callable]: Dictionary of all chains
    """
    chains = {
        "reasoning": chain_factory.create_reasoning_chain(),
        "analysis": chain_factory.create_analysis_chain(),
        "validation": chain_factory.create_validation_chain(),
        "tool_selection": chain_factory.create_tool_selection_chain(),
        "error_recovery": chain_factory.create_error_recovery_chain(),
        "response": chain_factory.create_response_chain(),
    }
    
    logger.info(f"Created {len(chains)} chains for ReAct agent")
    return chains
