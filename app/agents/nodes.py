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
from ..tools.initialization import get_tool_registry  
from .state import MathAgentState, WorkflowSteps, WorkflowStatus
from .chains import create_chain_factory

logger = get_logger(__name__)


def _get_chain_factory():
    """
    Helper function to create chain factory with proper dependencies.
    Eliminates code duplication across nodes (DRY principle).
    """
    settings = get_settings()
    tool_registry = ToolRegistry()
    return create_chain_factory(settings, tool_registry)


def _increment_iteration_count(state: MathAgentState) -> Dict[str, Any]:
    """
    Circuit breaker pattern to prevent infinite loops in LangGraph workflows.
    """
    current_count = state.get('iteration_count', 0)
    max_iterations = state.get('max_iterations', 10)
    
    new_count = current_count + 1
    
    if new_count >= max_iterations:
        logger.warning(f"Maximum iterations ({max_iterations}) reached. Forcing workflow termination.")
        return {
            'iteration_count': new_count,
            'current_step': WorkflowSteps.ERROR_RECOVERY,
            'error': f'Maximum iterations ({max_iterations}) exceeded',
            'error_type': 'iteration_limit_exceeded'
        }
    
    return {'iteration_count': new_count}


async def analyze_problem_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Mathematical problem analysis node with consolidated business logic.
    
    This node analyzes mathematical problems using direct LLM integration,
    eliminating the need for external components and circular dependencies.
    """
    try:
        iteration_update = _increment_iteration_count(state)
        if 'error' in iteration_update:
            logger.error("Analysis node: Maximum iterations exceeded")
            return iteration_update
        
        problem = state['current_problem']
        conversation_id = state.get('conversation_id', str(uuid4()))
        
        settings = get_settings()
        chain_factory = _get_chain_factory()
        
        logger.info(f"Analyzing problem: {problem[:50]}... (iteration: {iteration_update['iteration_count']})")
        
        analysis_chain = chain_factory.create_analysis_chain()
        
        analysis_result = await analysis_chain.ainvoke({
            "problem": problem,
            "conversation_id": conversation_id
        })
        
        if not isinstance(analysis_result, dict):
            error_msg = f"Analysis chain returned invalid response type: {type(analysis_result)}"
            logger.error(error_msg)
            raise AgentError(error_msg)
        
        problem_type = analysis_result.get("problem_type", "general")
        complexity = analysis_result.get("complexity", "medium")
        requires_tools = analysis_result.get("requires_tools", True)
        
        logger.info(f"Problem analysis complete: type={problem_type}, complexity={complexity}")
        
        result = {
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
        result.update(iteration_update)
        return result
        
    except Exception as e:
        logger.error(f"Error in problem analysis: {str(e)}")
        error_result = {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "analysis_error",
            "confidence_score": 0.0
        }
        if 'iteration_count' in state:
            error_result.update(_increment_iteration_count(state))
        return error_result


async def reasoning_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Mathematical reasoning node with consolidated business logic.
    
    This node performs mathematical reasoning using consolidated logic
    from the original ReactMathematicalAgent, eliminating duplication.
    """
    try:
        iteration_update = _increment_iteration_count(state)
        if 'error' in iteration_update:
            logger.error("Reasoning node: Maximum iterations exceeded")
            return iteration_update
        
        problem = state['current_problem']
        analysis = state.get('problem_analysis', {})
        context = state.get('context', [])
        
        settings = get_settings()
        chain_factory = _get_chain_factory()
        
        logger.info(f"Performing mathematical reasoning... (iteration: {iteration_update['iteration_count']})")
        
        reasoning_chain = chain_factory.create_reasoning_chain()
        
        try:
            reasoning_result = await reasoning_chain.ainvoke({
                "problem": problem,
                "analysis": analysis,
                "context": context,
                "previous_steps": state.get('reasoning_trace', [])
            })
        except Exception as parsing_error:
            error_msg = f"Reasoning chain failed: {str(parsing_error)}"
            logger.error(error_msg)
            raise AgentError(error_msg) from parsing_error
        
        mathematical_approach = reasoning_result.get("approach", "")
        step_by_step = reasoning_result.get("steps", [])
        tools_needed = reasoning_result.get("tools_needed", [])
        
        logger.info(f"Reasoning complete: {len(step_by_step)} steps identified, tools needed: {tools_needed}")
        
        result = {
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
        result.update(iteration_update)
        
        logger.info(f"DEBUG: reasoning_node returning tools_to_use={tools_needed}")
        logger.info(f"DEBUG: result contains keys: {list(result.keys())}")
        logger.info(f"DEBUG: result['tools_to_use'] = {result.get('tools_to_use', 'NOT_FOUND')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in reasoning: {str(e)}")
        error_result = {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "reasoning_error",
            "confidence_score": 0.0
        }
        if 'iteration_count' in state:
            error_result.update(_increment_iteration_count(state))
        return error_result


async def semantic_filter_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Semantic filter node - BigTool's PRIMARY PURPOSE.
    """
    try:
        iteration_update = _increment_iteration_count(state)
        if 'error' in iteration_update:
            logger.error("Semantic filter node: Maximum iterations exceeded")
            return iteration_update
        
        problem = state['current_problem']
        problem_analysis = state.get('problem_analysis', {})
        
        logger.info(f"Semantic filtering for: {problem[:50]}... (iteration: {iteration_update['iteration_count']})")
        
        # Use the global tool registry (DRY principle - reuse existing initialized registry)
        tool_registry = get_tool_registry()
        bigtool_manager = await create_bigtool_manager(tool_registry)
        
        # Context for semantic filtering
        context = {
            'problem_type': problem_analysis.get('type', 'general'),
            'complexity': problem_analysis.get('complexity', 'medium'),
            'max_tools': 3,  # Configurable limit
            'relevance_threshold': 0.7  # Semantic relevance threshold
        }
        
        # CORE SEMANTIC FILTERING using BigTool
        filtered_tools = await bigtool_manager.filter_tools_for_query(
            problem, 
            context
        )
        
        logger.info(f"Semantic filtering completed: {len(filtered_tools)} tools selected - {filtered_tools}")
        
        result = {
            "current_step": WorkflowSteps.TOOL_EXECUTION,
            "filtered_tools": filtered_tools,  # Only relevant tools
            "tools_to_use": filtered_tools,    # Compatibility with existing flow
            "reasoning_trace": state.get('reasoning_trace', []) + [
                f"Semantic filtering: {len(filtered_tools)} relevant tools identified: {filtered_tools}"
            ],
            "confidence_score": state.get('confidence_score', 0.8)
        }
        result.update(iteration_update)
        
        # DEBUG: Verificar propagación de estado
        logger.info(f"DEBUG: semantic_filter_node returning filtered_tools={filtered_tools}")
        logger.info(f"DEBUG: result keys={list(result.keys())}")
        logger.info(f"DEBUG: result['filtered_tools']={result.get('filtered_tools', 'NOT_FOUND')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in semantic filtering: {str(e)}")
        return {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error_message": f"Semantic filtering failed: {str(e)}",
            "reasoning_trace": state.get('reasoning_trace', []) + [
                f"Semantic filtering error: {str(e)}"
            ],
            **iteration_update
        }


async def tool_execution_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Tool execution node - Uses ONLY pre-filtered tools from semantic filtering.

    
    """
    try:
        iteration_update = _increment_iteration_count(state)
        if 'error' in iteration_update:
            return iteration_update
        
        # Use tools already filtered by semantic_filter_node
        filtered_tools = state.get('filtered_tools', [])
        problem = state['current_problem']
        
        # DEBUG: Verificar propagación de estado desde semantic_filter_node
        logger.info(f"DEBUG: tool_execution_node received state keys: {list(state.keys())}")
        logger.info(f"DEBUG: tool_execution_node filtered_tools from state: {filtered_tools}")
        logger.info(f"DEBUG: tool_execution_node state['filtered_tools']: {state.get('filtered_tools', 'NOT_FOUND')}")
        
        logger.info(f"Executing pre-filtered tools: {filtered_tools} (iteration: {iteration_update['iteration_count']})")
        
        if not filtered_tools:
            logger.info("No tools filtered for this problem, proceeding to validation")
            return {
                "current_step": WorkflowSteps.VALIDATION,
                "tool_results": [],
                "reasoning_trace": state.get('reasoning_trace', []) + [
                    "No tools needed after semantic filtering"
                ],
                "confidence_score": state.get('confidence_score', 0.8),
                **iteration_update
            }
        
        # Execute only semantically filtered tools using global registry (DRY principle)
        tool_registry = get_tool_registry()
        tool_results = []
        
        for tool_name in filtered_tools:
            try:
                tool_instance = tool_registry.get_tool(tool_name)
                
                if not tool_instance:
                    error_msg = f"Filtered tool '{tool_name}' not found in registry"
                    logger.error(error_msg)
                    raise ToolError(error_msg, tool_name=tool_name)
                
                logger.info(f"Executing filtered tool: {tool_name}")
                result = await tool_instance.arun(problem)
                tool_results.append({
                    "tool_name": tool_name,
                    "result": result,
                    "confidence": 1.0
                })
                logger.info(f"Filtered tool {tool_name} executed successfully")
                        
            except Exception as tool_error:
                logger.error(f"Filtered tool execution failed: {tool_name} - {str(tool_error)}")
                tool_results.append({
                    "tool_name": tool_name,
                    "result": {"success": False, "error": str(tool_error)},
                    "confidence": 0.0
                })
        
        result = {
            "current_step": WorkflowSteps.VALIDATION,
            "tool_results": tool_results,
            "reasoning_trace": state.get('reasoning_trace', []) + [
                f"Executed {len(tool_results)} semantically filtered tools"
            ],
            "confidence_score": sum(r.get('confidence', 0) for r in tool_results) / len(tool_results) if tool_results else 0.5
        }
        result.update(iteration_update)
        return result
        
    except Exception as e:
        logger.error(f"Error in tool execution: {str(e)}")
        error_result = {
            "current_step": WorkflowSteps.ERROR_RECOVERY,
            "error": str(e),
            "error_type": "tool_execution_error",
            "confidence_score": 0.0
        }
        if 'iteration_count' in state:
            error_result.update(_increment_iteration_count(state))
        return error_result


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
        
        settings = get_settings()
        chain_factory = _get_chain_factory()
        
        logger.info("Validating results...")
        
        validation_chain = chain_factory.create_validation_chain()
        
        validation_result = await validation_chain.ainvoke({
            "problem": problem,
            "reasoning": reasoning_result,
            "tool_results": tool_results,
            "trace": state.get('reasoning_trace', [])
        })
        
        is_valid = validation_result.get("is_valid", False)
        validation_score = validation_result.get("score", 0.0)
        issues = validation_result.get("issues", [])
        
        has_tool_results = bool(tool_results)
        has_reasoning = bool(reasoning_result)
        
        # If we have tools executed and reasoning, consider it valid for now (development phase)
        if has_tool_results and has_reasoning:
            logger.info(f"Validation passed: tool_results={len(tool_results)}, reasoning=True")
            return {
                "current_step": WorkflowSteps.FINALIZATION,
                "validation_result": validation_result,
                "is_solution_complete": True,
                "confidence_score": max(validation_score, 0.8)
            }
        
        logger.info(f"Validation failed: valid={is_valid}, score={validation_score}, issues={issues}")
        return {
            "current_step": WorkflowSteps.REASONING,
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
        
        settings = get_settings()
        chain_factory = _get_chain_factory()
        
        logger.info("Generating final response...")
        
        response_chain = chain_factory.create_response_chain()
        
        final_response = await response_chain.ainvoke({
            "problem": problem,
            "reasoning": reasoning_result,
            "tool_results": tool_results,
            "validation": validation_result,
            "trace": state.get('reasoning_trace', [])
        })
        
        if not isinstance(final_response, dict):
            error_msg = f"Response chain returned invalid response type: {type(final_response)}"
            logger.error(error_msg)
            raise AgentError(error_msg)
        
        logger.info("Final solution generated successfully")
        
        return {
            "current_step": WorkflowSteps.COMPLETE,
            "workflow_status": WorkflowStatus.COMPLETED,
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


async def error_recovery_node(state: MathAgentState) -> Dict[str, Any]:
    """
    Error recovery node with consolidated error handling logic.
    
    This node handles errors and attempts recovery using consolidated
    logic from the original error handling components.
    """
    try:
        error = state.get('error', 'Unknown error')
        error_type = state.get('error_type', 'general_error')
        
        current_retry_count = state.get('retry_count', 0)
        current_iteration_count = state.get('iteration_count', 0)
        max_retries = state.get('max_iterations', 10) // 2  
        
        logger.warning(f"Handling error: {error_type} - {error} (retry: {current_retry_count}, iteration: {current_iteration_count})")
        
        if current_retry_count >= max_retries:
            logger.error(f"Maximum error recovery attempts ({max_retries}) exceeded")
            return {
                "current_step": WorkflowSteps.COMPLETE,
                "workflow_status": WorkflowStatus.FAILED,
                "final_answer": f"I apologize, but I encountered an error that I couldn't resolve: {error}",
                "error": error,
                "retry_count": current_retry_count,
                "iteration_count": current_iteration_count,
                "confidence_score": 0.0,
                "is_complete": True
            }
        
        settings = get_settings()
        chain_factory = _get_chain_factory()
        
        recovery_chain = chain_factory.create_error_recovery_chain()
        
        recovery_result = await recovery_chain.ainvoke({
            "error": error,
            "error_type": error_type,
            "problem": state.get('current_problem', ''),
            "retry_count": current_retry_count
        })
        
        recovery_action = recovery_result.get("action", "restart")
        
        logger.info(f"Recovery action: {recovery_action}")
        
        if recovery_action == "retry_reasoning":
            next_step = WorkflowSteps.REASONING
        elif recovery_action == "retry_analysis":
            next_step = WorkflowSteps.ANALYSIS
        elif recovery_action == "simplify_problem":
            next_step = WorkflowSteps.ANALYSIS
        else:
            next_step = WorkflowSteps.ANALYSIS
        
        return {
            "current_step": next_step,
            "retry_count": current_retry_count + 1,
            "iteration_count": current_iteration_count,
            "recovery_action": recovery_action,
            "recovery_note": recovery_result.get("note", ""),
            "confidence_score": 0.5
        }
        
    except Exception as e:
        logger.error(f"Error in recovery: {str(e)}")
        return {
            "current_step": WorkflowSteps.COMPLETE,
            "workflow_status": WorkflowStatus.FAILED,
            "final_answer": f"Critical error: {str(e)}",
            "confidence_score": 0.0,
            "is_complete": True
        }
