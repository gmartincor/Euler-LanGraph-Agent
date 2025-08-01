"""Professional LangGraph Workflow Nodes - Real Implementation.

This module contains the actual workflow node implementations using the extracted
core business logic components, eliminating circular dependencies and code duplication.

Key Design Principles Applied:
- Composition over Inheritance: Uses core components via dependency injection
- Single Responsibility: Each node has one clear workflow responsibility
- Zero Circular Dependencies: No backwards references to agent classes
- Professional Error Handling: Comprehensive exception management
- Real Business Logic: Actual implementations, not delegation wrappers

Architecture Benefits:
- No Code Duplication: Uses core business logic components
- Testable: Each node can be tested independently with mocked dependencies
- Maintainable: Clear separation between workflow and business logic
- Professional Quality: Production-ready implementations
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

from ...core.logging import get_logger, log_function_call
from ...core.exceptions import AgentError, ToolError, ValidationError
from ..state import MathAgentState, WorkflowSteps
from ..core.mathematical_reasoner import MathematicalReasoner
from ..core.tool_orchestrator import ToolOrchestrator
from ..core.state_manager import StateManager

logger = get_logger(__name__)


# === Professional Workflow Nodes with Real Logic ===

@log_function_call(logger)
async def analyze_problem_node(
    state: MathAgentState,
    reasoner: MathematicalReasoner,
    state_manager: StateManager
) -> Dict[str, Any]:
    """
    Mathematical problem analysis workflow node.
    
    This node implements actual problem analysis using the MathematicalReasoner
    component, providing real business logic without circular dependencies.
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        
        logger.info(f"Starting problem analysis for: {problem[:50]}...")
        
        # Perform actual problem analysis using reasoning engine
        analysis_results = await reasoner.analyze_problem(
            problem=problem,
            mathematical_context=state['mathematical_context']
        )
        
        # Update mathematical context with analysis results
        state_updates = {
            'mathematical_context': {
                **state['mathematical_context'],
                'problem_type': analysis_results.get('problem_type', 'unknown'),
                'complexity': analysis_results.get('complexity', 'medium'),
                'strategy': analysis_results.get('strategy', 'analytical')
            },
            'confidence_score': analysis_results.get('confidence', 0.5)
        }
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Problem analysis failed: {e}", exc_info=True)
        raise AgentError(f"Problem analysis node failed: {str(e)}") from e


@log_function_call(logger)  
async def reasoning_node(
    state: MathAgentState,
    reasoner: MathematicalReasoner,
    state_manager: StateManager
) -> Dict[str, Any]:
    """Mathematical reasoning workflow node with real logic."""
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        
        # Get available tools for reasoning
        available_tools = state.get('selected_tools', ['integral_tool', 'analysis_tool', 'plot_tool'])
        
        # Perform actual reasoning using reasoning engine
        reasoning_results = await reasoner.perform_reasoning(
            problem=problem,
            analysis_results=state['mathematical_context'],
            available_tools=available_tools,
            previous_attempts=state.get('reasoning_chain', [])
        )
        
        # Update state with reasoning results
        state_updates = {
            'reasoning_chain': state.get('reasoning_chain', []) + [reasoning_results],
            'selected_tools': reasoning_results.get('tool_plan', available_tools),
            'confidence_score': reasoning_results.get('confidence', 0.5)
        }
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Mathematical reasoning failed: {e}", exc_info=True)
        raise AgentError(f"Reasoning node failed: {str(e)}") from e


@log_function_call(logger)
async def tool_execution_node(
    state: MathAgentState,
    tool_orchestrator: ToolOrchestrator,
    state_manager: StateManager
) -> Dict[str, Any]:
    """Tool execution workflow node with real orchestration logic."""
    try:
        selected_tools = state.get('selected_tools', ['integral_tool'])
        problem = state['current_problem']
        
        # Prepare tool parameters
        tool_parameters = {}
        for tool_name in selected_tools:
            if tool_name == 'integral_tool':
                tool_parameters[tool_name] = {
                    'expression': problem,
                    'variable': 'x',
                    'method': 'symbolic'
                }
            else:
                tool_parameters[tool_name] = {'input': problem}
        
        # Execute tools using orchestrator
        execution_results = await tool_orchestrator.execute_tools_parallel(
            tool_names=selected_tools,
            tool_parameters=tool_parameters,
            execution_strategy='parallel'
        )
        
        # Update state with results
        state_updates = {
            'tool_results': execution_results['results'],
            'confidence_score': min(state.get('confidence_score', 0.5) * 1.2, 1.0)
        }
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        raise AgentError(f"Tool execution node failed: {str(e)}") from e


@log_function_call(logger)
async def validation_node(
    state: MathAgentState,
    reasoner: MathematicalReasoner,
    state_manager: StateManager
) -> Dict[str, Any]:
    """Result validation workflow node with real validation logic."""
    try:
        problem = state['current_problem']
        tool_results = state.get('tool_results', {})
        reasoning_chain = state.get('reasoning_chain', [])
        
        # Perform result validation using reasoning engine
        validation_results = await reasoner.validate_results(
            problem=problem,
            reasoning_steps=reasoning_chain,
            tool_results=tool_results,
            expected_outcome={}
        )
        
        # Update state based on validation
        is_valid = validation_results.get('is_valid', False)
        confidence = validation_results.get('confidence', 0.5)
        
        state_updates = {
            'confidence_score': confidence
        }
        
        # Set final answer if validation passed
        if is_valid and confidence > 0.7:
            state_updates['final_answer'] = tool_results
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Result validation failed: {e}", exc_info=True)
        raise AgentError(f"Validation node failed: {str(e)}") from e


@log_function_call(logger)
async def final_response_node(
    state: MathAgentState,
    state_manager: StateManager
) -> Dict[str, Any]:
    """Final response generation workflow node."""
    try:
        problem = state['current_problem']
        final_answer = state.get('final_answer', state.get('tool_results', {}))
        confidence_score = state.get('confidence_score', 0.5)
        
        # Generate comprehensive response
        response = {
            'problem_statement': problem,
            'final_answer': final_answer,
            'confidence_score': confidence_score,
            'tools_used': list(state.get('tool_results', {}).keys()),
            'status': 'COMPLETED'
        }
        
        state_updates = {
            'workflow_status': 'COMPLETED',
            'final_answer': response
        }
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Final response generation failed: {e}", exc_info=True)
        raise AgentError(f"Final response node failed: {str(e)}") from e


@log_function_call(logger)
async def error_recovery_node(
    state: MathAgentState,
    reasoner: MathematicalReasoner,
    state_manager: StateManager
) -> Dict[str, Any]:
    """Error recovery workflow node with intelligent recovery."""
    try:
        problem = state['current_problem']
        error_count = state.get('error_count', 0)
        
        if error_count >= 3:
            # Generate fallback response
            fallback_response = {
                'problem_statement': problem,
                'status': 'PARTIAL_SOLUTION',
                'message': 'Unable to fully solve the problem after multiple attempts.',
                'partial_results': state.get('tool_results', {})
            }
            
            state_updates = {
                'workflow_status': 'COMPLETED_WITH_ERRORS',
                'final_answer': fallback_response,
                'confidence_score': 0.2
            }
        else:
            # Attempt recovery
            state_updates = {
                'error_count': error_count + 1,
                'selected_tools': ['analysis_tool'],  # Fallback to basic tool
                'confidence_score': 0.4
            }
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Error recovery failed: {e}", exc_info=True)
        raise AgentError(f"Error recovery node failed: {str(e)}") from e
