"""Professional LangGraph Workflow Nodes - Unified Architecture.

This module contains ALL mathematical reasoning business logic consolidated into
pure, stateless functions. Eliminates circular dependencies and code duplication
following DRY, KISS, YAGNI principles.

Key Design Principles Applied:
- Pure Functions: No side effects, fully testable
- Single Responsibility: Each node has one clear workflow responsibility  
- Zero Circular Dependencies: No imports from agents or graph modules
- Consolidated Logic: All business logic in one place (DRY)
- Professional Error Handling: Comprehensive exception management

Architecture Benefits:
- Zero Code Duplication: Single source of truth for business logic
- Maximum Testability: Pure functions with clear interfaces
- High Maintainability: One place to change business logic
- Production Ready: Professional error handling and logging
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from uuid import uuid4

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ..core.logging import get_logger, log_function_call
from ..core.exceptions import AgentError, ToolError, ValidationError
from ..core.config import get_settings
from ..core.bigtool_setup import create_bigtool_manager
from ..tools.registry import ToolRegistry
from .state import MathAgentState, WorkflowSteps, WorkflowStatus
from .chains import create_chain_factory

logger = get_logger(__name__)


# === Consolidated Mathematical Reasoning Nodes ===

@log_function_call(logger)
async def analyze_problem_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Mathematical problem analysis node with consolidated business logic.
    
    This node analyzes mathematical problems using direct LLM integration,
    eliminating the need for external components and circular dependencies.
    """
    try:
        problem = state['current_problem']
        conversation_id = state.get('conversation_id', str(uuid4()))
        
        # Get settings and create chain factory
        settings = get_settings()
        chain_factory = create_chain_factory()
        
        logger.info(f"Analyzing problem: {problem[:50]}...")
        
        # Create analysis chain with consolidated logic
        analysis_chain = chain_factory.create_analysis_chain()
        
        # Perform analysis
        analysis_result = await analysis_chain.ainvoke({
            "problem": problem,
            "conversation_id": conversation_id
        })
        
        # Extract problem type and complexity
        problem_type = analysis_result.get("problem_type", "general")
        complexity = analysis_result.get("complexity", "medium")
        requires_tools = analysis_result.get("requires_tools", True)
        
        logger.info(f"Problem analysis complete: type={problem_type}, complexity={complexity}")
        
        return {
            "current_step": WorkflowSteps.ANALYSIS,
            "problem_analysis": {
                "type": problem_type,
                "complexity": complexity,
                "requires_tools": requires_tools,
                "description": analysis_result.get("description", ""),
                "approach": analysis_result.get("approach", "")
            },
            "reasoning_trace": [f"Problem analyzed: {problem_type} - {complexity}"],
            "confidence_score": analysis_result.get("confidence", 0.8)
        }
        
    except Exception as e:
        logger.error(f"Error in problem analysis: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "analysis_error",
            "confidence_score": 0.0
        }


@log_function_call(logger)
async def reasoning_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Mathematical reasoning node with consolidated business logic.
    
    This node performs mathematical reasoning using consolidated logic
    from the original ReactMathematicalAgent, eliminating duplication.
    """
    try:
        problem = state['current_problem']
        analysis = state.get('problem_analysis', {})
        context = state.get('context', [])
        
        # Get settings and create chain factory
        settings = get_settings()
        chain_factory = create_chain_factory()
        
        logger.info("Performing mathematical reasoning...")
        
        # Create reasoning chain
        reasoning_chain = chain_factory.create_reasoning_chain()
        
        # Perform reasoning with context
        reasoning_result = await reasoning_chain.ainvoke({
            "problem": problem,
            "analysis": analysis,
            "context": context,
            "previous_steps": state.get('reasoning_trace', [])
        })
        
        # Extract reasoning components
        mathematical_approach = reasoning_result.get("approach", "")
        step_by_step = reasoning_result.get("steps", [])
        tools_needed = reasoning_result.get("tools_needed", [])
        
        logger.info(f"Reasoning complete: {len(step_by_step)} steps identified")
        
        return {
            "current_step": WorkflowSteps.REASONING,
            "reasoning_result": {
                "approach": mathematical_approach,
                "steps": step_by_step,
                "tools_needed": tools_needed,
                "confidence": reasoning_result.get("confidence", 0.8)
            },
            "tools_to_use": tools_needed,
            "reasoning_trace": state.get('reasoning_trace', []) + [f"Reasoning: {mathematical_approach}"],
            "confidence_score": reasoning_result.get("confidence", 0.8)
        }
        
    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "reasoning_error",
            "confidence_score": 0.0
        }


@log_function_call(logger)
async def tool_execution_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Tool execution node with consolidated BigTool integration.
    
    This node executes mathematical tools using BigTool for intelligent
    tool selection, consolidating logic from tool_orchestrator.
    """
    try:
        tools_needed = state.get('tools_to_use', [])
        problem = state['current_problem']
        
        if not tools_needed:
            logger.info("No tools needed, skipping tool execution")
            return {
                "current_step": WorkflowSteps.VALIDATION,
                "tool_results": [],
                "confidence_score": state.get('confidence_score', 0.8)
            }
        
        # Initialize BigTool and ToolRegistry
        bigtool_manager = create_bigtool_manager()
        tool_registry = ToolRegistry()
        
        logger.info(f"Executing tools: {tools_needed}")
        
        tool_results = []
        
        for tool_name in tools_needed:
            try:
                # Use BigTool for intelligent tool selection
                recommended_tools = await bigtool_manager.search_tools(
                    query=f"{tool_name} {problem}",
                    top_k=3
                )
                
                if recommended_tools:
                    # Execute the most relevant tool
                    best_tool = recommended_tools[0]
                    tool_instance = tool_registry.get_tool(best_tool.name)
                    
                    if tool_instance:
                        result = await tool_instance.arun(problem)
                        tool_results.append({
                            "tool_name": best_tool.name,
                            "result": result,
                            "confidence": best_tool.similarity_score
                        })
                        logger.info(f"Tool {best_tool.name} executed successfully")
                    else:
                        logger.warning(f"Tool {best_tool.name} not found in registry")
                else:
                    logger.warning(f"No tools found for: {tool_name}")
                    
            except Exception as tool_error:
                logger.error(f"Error executing tool {tool_name}: {str(tool_error)}")
                tool_results.append({
                    "tool_name": tool_name,
                    "error": str(tool_error),
                    "confidence": 0.0
                })
        
        return {
            "current_step": WorkflowSteps.VALIDATION,
            "tool_results": tool_results,
            "reasoning_trace": state.get('reasoning_trace', []) + [f"Tools executed: {len(tool_results)} results"],
            "confidence_score": sum(r.get('confidence', 0) for r in tool_results) / len(tool_results) if tool_results else 0.5
        }
        
    except Exception as e:
        logger.error(f"Error in tool execution: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "tool_execution_error",
            "confidence_score": 0.0
        }


@log_function_call(logger)
async def validation_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Result validation node with consolidated validation logic.
    
    This node validates mathematical results using consolidated logic
    from the original validation components.
    """
    try:
        tool_results = state.get('tool_results', [])
        reasoning_result = state.get('reasoning_result', {})
        problem = state['current_problem']
        
        # Get settings and create chain factory
        settings = get_settings()
        chain_factory = create_chain_factory()
        
        logger.info("Validating results...")
        
        # Create validation chain
        validation_chain = chain_factory.create_validation_chain()
        
        # Perform validation
        validation_result = await validation_chain.ainvoke({
            "problem": problem,
            "reasoning": reasoning_result,
            "tool_results": tool_results,
            "trace": state.get('reasoning_trace', [])
        })
        
        is_valid = validation_result.get("is_valid", False)
        validation_score = validation_result.get("score", 0.0)
        issues = validation_result.get("issues", [])
        
        logger.info(f"Validation complete: valid={is_valid}, score={validation_score}")
        
        if is_valid and validation_score >= 0.7:
            return {
                "current_step": WorkflowSteps.FINALIZATION,
                "validation_result": validation_result,
                "is_solution_complete": True,
                "confidence_score": validation_score
            }
        else:
            return {
                "current_step": WorkflowSteps.REASONING,  # Back to reasoning
                "validation_result": validation_result,
                "is_solution_complete": False,
                "needs_improvement": True,
                "validation_issues": issues,
                "confidence_score": validation_score
            }
        
    except Exception as e:
        logger.error(f"Error in validation: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "validation_error",
            "confidence_score": 0.0
        }


@log_function_call(logger)
async def finalization_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Solution finalization node with consolidated response generation.
    
    This node generates the final mathematical solution using consolidated
    logic from the original response generation components.
    """
    try:
        problem = state['current_problem']
        reasoning_result = state.get('reasoning_result', {})
        tool_results = state.get('tool_results', [])
        validation_result = state.get('validation_result', {})
        
        # Get settings and create chain factory
        settings = get_settings()
        chain_factory = create_chain_factory()
        
        logger.info("Generating final response...")
        
        # Create response chain
        response_chain = chain_factory.create_response_chain()
        
        # Generate final response
        final_response = await response_chain.ainvoke({
            "problem": problem,
            "reasoning": reasoning_result,
            "tool_results": tool_results,
            "validation": validation_result,
            "trace": state.get('reasoning_trace', [])
        })
        
        logger.info("Final solution generated successfully")
        
        return {
            "current_step": WorkflowSteps.COMPLETE,
            "status": WorkflowStatus.COMPLETED,
            "final_answer": final_response.get("answer", ""),
            "solution_steps": final_response.get("steps", []),
            "explanation": final_response.get("explanation", ""),
            "confidence_score": final_response.get("confidence", 0.8),
            "is_complete": True
        }
        
    except Exception as e:
        logger.error(f"Error in finalization: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "finalization_error",
            "confidence_score": 0.0
        }


@log_function_call(logger)
async def error_recovery_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Error recovery node with consolidated error handling logic.
    
    This node handles errors and attempts recovery using consolidated
    logic from the original error handling components.
    """
    try:
        error = state.get('error', 'Unknown error')
        error_type = state.get('error_type', 'general_error')
        retry_count = state.get('retry_count', 0)
        
        logger.warning(f"Handling error: {error_type} - {error}")
        
        # Get settings and create chain factory
        settings = get_settings()
        chain_factory = create_chain_factory()
        
        # Maximum retry attempts
        max_retries = 3
        
        if retry_count >= max_retries:
            logger.error(f"Maximum retries ({max_retries}) exceeded")
            return {
                "current_step": WorkflowSteps.COMPLETE,
                "status": WorkflowStatus.FAILED,
                "final_answer": f"I apologize, but I encountered an error that I couldn't resolve: {error}",
                "error": error,
                "retry_count": retry_count,
                "confidence_score": 0.0,
                "is_complete": True
            }
        
        # Create error recovery chain
        recovery_chain = chain_factory.create_error_recovery_chain()
        
        # Attempt recovery
        recovery_result = await recovery_chain.ainvoke({
            "error": error,
            "error_type": error_type,
            "problem": state.get('current_problem', ''),
            "retry_count": retry_count
        })
        
        recovery_action = recovery_result.get("action", "restart")
        
        logger.info(f"Recovery action: {recovery_action}")
        
        # Determine next step based on recovery action
        if recovery_action == "retry_reasoning":
            next_step = WorkflowSteps.REASONING
        elif recovery_action == "retry_analysis":
            next_step = WorkflowSteps.ANALYSIS
        elif recovery_action == "simplify_problem":
            next_step = WorkflowSteps.ANALYSIS
        else:
            next_step = WorkflowSteps.ANALYSIS  # Default to restart
        
        return {
            "current_step": next_step,
            "retry_count": retry_count + 1,
            "recovery_action": recovery_action,
            "recovery_note": recovery_result.get("note", ""),
            "confidence_score": 0.5  # Medium confidence after recovery
        }
        
    except Exception as e:
        logger.error(f"Error in recovery: {str(e)}")
        return {
            "current_step": WorkflowSteps.COMPLETE,
            "status": WorkflowStatus.FAILED,
            "final_answer": f"Critical error: {str(e)}",
            "confidence_score": 0.0,
            "is_complete": True
        }
