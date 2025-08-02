"""Simplified chain factories for mathematical ReAct agent.

This module provides factory functions for creating different types of chains
used in the ReAct reasoning process. Simplified to remove external dependencies
and focus on core functionality.
"""

from typing import Any, Dict, List, Optional, Callable

from ..core.config import Settings
from ..core.logging import get_logger, log_function_call
from ..tools.registry import ToolRegistry

logger = get_logger(__name__)


class ChainFactory:
    """
    Simplified factory class for creating specialized chains for mathematical ReAct agent.
    
    This factory focuses on core functionality without external dependencies.
    """
    
    def __init__(
        self,
        settings: Settings,
        tool_registry: ToolRegistry
    ):
        """
        Initialize chain factory.
        
        Args:
            settings: Application settings
            tool_registry: Existing tool registry
        """
        self.settings = settings
        self.tool_registry = tool_registry
        
        logger.info("ChainFactory initialized with existing infrastructure")
    
    def create_reasoning_chain(self) -> Callable:
        """
        Create mathematical reasoning chain.
        
        Returns:
            Callable: Reasoning function
        """
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
        
        logger.info("Mathematical reasoning chain created")
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
                
                # Simple rule-based analysis for now
                problem_lower = problem.lower()
                
                # Determine problem type
                if "integral" in problem_lower or "∫" in problem:
                    problem_type = "integral"
                    complexity = "medium"
                    requires_tools = True
                elif "derivative" in problem_lower or "d/dx" in problem_lower:
                    problem_type = "derivative"
                    complexity = "low"
                    requires_tools = True
                elif "plot" in problem_lower or "graph" in problem_lower:
                    problem_type = "visualization"
                    complexity = "low" 
                    requires_tools = True
                else:
                    problem_type = "general"
                    complexity = "medium"
                    requires_tools = True
                
                return {
                    "problem_type": problem_type,
                    "complexity": complexity,
                    "requires_tools": requires_tools,
                    "description": f"Mathematical {problem_type} problem",
                    "approach": f"Use {problem_type} calculation methods",
                    "confidence": 0.8
                }
                
            except Exception as e:
                logger.error(f"Error in problem analysis: {e}")
                return {
                    "problem_type": "general",
                    "complexity": "unknown",
                    "requires_tools": True,
                    "description": "Error analyzing problem",
                    "approach": "Basic mathematical approach",
                    "confidence": 0.5
                }
        
        logger.info("Problem analysis chain created")
        return analyze_problem
    
    def create_validation_chain(self) -> Callable:
        """
        Create result validation and reflection chain.
        
        Returns:
            Callable: Validation function
        """
        async def validate_results(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Validate mathematical results and reasoning."""
            try:
                problem = input_data.get("problem", "")
                reasoning = input_data.get("reasoning", {})
                tool_results = input_data.get("tool_results", [])
                trace = input_data.get("trace", [])
                
                # Simple validation logic
                has_results = len(tool_results) > 0
                has_errors = any("error" in result for result in tool_results)
                
                if has_results and not has_errors:
                    is_valid = True
                    score = 0.9
                    issues = []
                elif has_results and has_errors:
                    is_valid = False
                    score = 0.4
                    issues = ["Some tool executions failed"]
                else:
                    is_valid = False
                    score = 0.3
                    issues = ["No results obtained"]
                
                # Check reasoning quality
                approach = reasoning.get("approach", "")
                if len(approach) < 10:
                    score -= 0.2
                    issues.append("Insufficient reasoning")
                
                return {
                    "is_valid": is_valid,
                    "score": score,
                    "issues": issues
                }
                
            except Exception as e:
                logger.error(f"Error in validation: {e}")
                return {
                    "is_valid": False,
                    "score": 0.0,
                    "issues": [f"Validation error: {str(e)}"]
                }
        
        logger.info("Validation chain created")
        return validate_results
    
    def create_error_recovery_chain(self) -> Callable:
        """
        Create error recovery chain.
        
        Returns:
            Callable: Error recovery function
        """
        async def recover_from_error(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Handle errors and determine recovery strategy."""
            try:
                error = input_data.get("error", "Unknown error")
                error_type = input_data.get("error_type", "general_error")
                retry_count = input_data.get("retry_count", 0)
                
                # Determine recovery action based on error type
                if "tool" in error_type.lower():
                    action = "retry_reasoning"
                    note = "Tool execution failed, trying alternative approach"
                elif "validation" in error_type.lower():
                    action = "simplify_problem"
                    note = "Validation failed, simplifying the problem"
                elif retry_count >= 2:
                    action = "restart"
                    note = "Multiple failures, restarting analysis"
                else:
                    action = "retry_analysis"
                    note = "General error, retrying from analysis"
                
                return {
                    "action": action,
                    "note": note
                }
                
            except Exception as e:
                logger.error(f"Error in recovery: {e}")
                return {
                    "action": "restart",
                    "note": f"Recovery failed: {str(e)}"
                }
        
        logger.info("Error recovery chain created")
        return recover_from_error
    
    def create_response_chain(self) -> Callable:
        """
        Create final response formatting chain.
        
        Returns:
            Callable: Response formatting function
        """
        async def format_response(input_data: Dict[str, Any]) -> Dict[str, Any]:
            """Format the final response with structured output."""
            try:
                problem = input_data.get("problem", "No problem specified")
                reasoning = input_data.get("reasoning", {})
                tool_results = input_data.get("tool_results", [])
                validation = input_data.get("validation", {})
                trace = input_data.get("trace", [])
                
                # Extract answer from tool results
                final_answer = "No answer calculated"
                if tool_results:
                    for result in tool_results:
                        if "result" in result and result["result"]:
                            final_answer = str(result["result"])
                            break
                
                # Build solution steps
                steps = []
                if reasoning.get("steps"):
                    steps.extend(reasoning["steps"])
                if trace:
                    steps.extend([f"Step: {step}" for step in trace[-3:]])  # Last 3 steps
                
                # Calculate confidence
                confidence = validation.get("score", reasoning.get("confidence", 0.7))
                
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
    tool_registry: ToolRegistry
) -> ChainFactory:
    """
    Factory function  to create ChainFactory instance.
    
    Args:
        settings: Application settings
        tool_registry: Existing tool registry
        
    Returns:
        ChainFactory: Initialized chain factory
    """
    return ChainFactory(settings, tool_registry)


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
        "error_recovery": chain_factory.create_error_recovery_chain(),
        "response": chain_factory.create_response_chain(),
    }
    
    logger.info(f"Created {len(chains)} chains for ReAct agent")
    return chains
