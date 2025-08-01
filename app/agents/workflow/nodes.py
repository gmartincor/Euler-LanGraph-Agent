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
    
    Responsibilities:
    - Problem classification and complexity assessment using reasoning engine
    - Mathematical context extraction and structuring
    - Strategy recommendation based on problem characteristics
    - State updates for analysis results
    
    Args:
        state: Current mathematical agent state
        reasoner: Mathematical reasoning engine for problem analysis
        state_manager: State manager for state transitions
        
    Returns:
        Dict[str, Any]: State updates containing analysis results
        
    Raises:
        AgentError: If problem analysis fails
        ValidationError: If problem cannot be analyzed
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
        updated_context = {
            **state['mathematical_context'],
            'problem_type': analysis_results.get('problem_type', 'unknown'),
            'complexity': analysis_results.get('complexity', 'medium'),
            'strategy': analysis_results.get('strategy', 'analytical')
        }
        
        # Prepare state updates
        state_updates = {
            'mathematical_context': updated_context,
            'confidence_score': analysis_results.get('confidence', 0.5),
            'execution_metadata': {
                **state['execution_metadata'],
                'analysis_completed': datetime.now().isoformat(),
                'problem_type': analysis_results.get('problem_type', 'unknown')
            }
        }
        
        # Update state through state manager
        await state_manager.transition_workflow_step(
            conversation_id,
            WorkflowSteps.TOOL_SELECTION,
            state_updates
        )
        
        logger.info(
            f"Problem analysis completed: {analysis_results.get('problem_type')} "
            f"(confidence: {analysis_results.get('confidence', 0):.2f})"
        )
        
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
    """
    Mathematical reasoning workflow node.
    
    This node implements step-by-step mathematical reasoning using the
    MathematicalReasoner component for actual problem-solving logic.
    
    Responsibilities:
    - Step-by-step problem decomposition using reasoning chains
    - Mathematical strategy formulation
    - Tool usage planning based on reasoning results
    - State updates for reasoning outcomes
    
    Args:
        state: Current mathematical agent state
        reasoner: Mathematical reasoning engine
        state_manager: State manager for state transitions
        
    Returns:
        Dict[str, Any]: State updates containing reasoning results
        
    Raises:
        AgentError: If reasoning process fails
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        analysis_results = {
            'problem_type': state['mathematical_context'].get('problem_type', 'unknown'),
            'complexity': state['mathematical_context'].get('complexity', 'medium'),
            'strategy': state['mathematical_context'].get('strategy', 'analytical')
        }
        
        logger.info(f"Starting mathematical reasoning for {analysis_results['problem_type']} problem")
        
        # Get available tools for reasoning
        available_tools = state.get('selected_tools', [])
        if not available_tools:
            # Default mathematical tools if none selected
            available_tools = ['integral_tool', 'analysis_tool', 'plot_tool']
        
        # Perform actual reasoning using reasoning engine
        reasoning_results = await reasoner.perform_reasoning(
            problem=problem,
            analysis_results=analysis_results,
            available_tools=available_tools,
            previous_attempts=state.get('reasoning_chain', [])
        )
        
        # Update reasoning chain with new results
        updated_chain = state.get('reasoning_chain', [])
        updated_chain.append({
            'timestamp': datetime.now().isoformat(),
            'reasoning_steps': reasoning_results.get('reasoning_steps', []),
            'confidence': reasoning_results.get('confidence', 0.5),
            'next_action': reasoning_results.get('next_action', 'use_tools')
        })
        
        # Prepare state updates
        state_updates = {
            'reasoning_chain': updated_chain,
            'selected_tools': reasoning_results.get('tool_plan', available_tools),
            'confidence_score': reasoning_results.get('confidence', 0.5),
            'execution_metadata': {
                **state['execution_metadata'],
                'reasoning_completed': datetime.now().isoformat(),
                'reasoning_steps_count': len(reasoning_results.get('reasoning_steps', []))
            }
        }
        
        # Determine next workflow step based on reasoning outcome
        next_action = reasoning_results.get('next_action', 'use_tools')
        if next_action == 'use_tools':
            next_step = WorkflowSteps.TOOL_EXECUTION
        elif next_action == 'finalize':
            next_step = WorkflowSteps.FINAL_RESPONSE
        else:
            next_step = WorkflowSteps.TOOL_SELECTION
        
        # Update state through state manager
        await state_manager.transition_workflow_step(
            conversation_id,
            next_step,
            state_updates
        )
        
        logger.info(
            f"Reasoning completed with {len(reasoning_results.get('reasoning_steps', []))} steps "
            f"→ {next_step.value}"
        )
        
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
    """
    Tool execution workflow node.
    
    This node implements intelligent tool execution using the ToolOrchestrator
    component for actual tool selection, execution, and result management.
    
    Responsibilities:
    - Tool selection based on problem analysis and reasoning
    - Parallel tool execution with performance optimization
    - Result validation and quality assessment
    - State updates for tool execution outcomes
    
    Args:
        state: Current mathematical agent state
        tool_orchestrator: Tool orchestration engine
        state_manager: State manager for state transitions
        
    Returns:
        Dict[str, Any]: State updates containing tool execution results
        
    Raises:
        ToolError: If tool execution fails
        AgentError: If tool orchestration fails
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        selected_tools = state.get('selected_tools', [])
        
        if not selected_tools:
            logger.warning("No tools selected for execution, using defaults")
            selected_tools = ['integral_tool', 'analysis_tool', 'plot_tool']
        
        logger.info(f"Starting tool execution: {selected_tools}")
        
        # Prepare tool parameters based on problem and context
        tool_parameters = {}
        mathematical_context = state['mathematical_context']
        
        for tool_name in selected_tools:
            if tool_name == 'integral_tool':
                tool_parameters[tool_name] = {
                    'expression': problem,
                    'variable': mathematical_context.get('variables', ['x'])[0] if mathematical_context.get('variables') else 'x',
                    'method': 'symbolic'
                }
            elif tool_name == 'plot_tool':
                tool_parameters[tool_name] = {
                    'expression': problem,
                    'x_range': mathematical_context.get('domain', [-10, 10]),
                    'plot_type': 'function'
                }
            elif tool_name == 'analysis_tool':
                tool_parameters[tool_name] = {
                    'expression': problem,
                    'analysis_type': 'comprehensive'
                }
            else:
                # Generic parameters for other tools
                tool_parameters[tool_name] = {
                    'input': problem,
                    'context': mathematical_context
                }
        
        # Execute tools using orchestrator
        execution_results = await tool_orchestrator.execute_tools_parallel(
            tool_names=selected_tools,
            tool_parameters=tool_parameters,
            execution_strategy='adaptive'
        )
        
        # Validate tool results
        validation_results = await tool_orchestrator.validate_tool_results(
            tool_results=execution_results['results'],
            expected_types={tool: dict for tool in selected_tools}
        )
        
        # Prepare state updates
        state_updates = {
            'tool_results': execution_results['results'],
            'execution_metadata': {
                **state['execution_metadata'],
                'tool_execution_completed': datetime.now().isoformat(),
                'tools_executed': execution_results['metadata']['tools_executed'],
                'tool_execution_time': execution_results['metadata']['execution_time'],
                'tool_success_rate': execution_results['metadata']['success_rate']
            }
        }
        
        # Update confidence based on tool results validation
        if validation_results['is_valid']:
            state_updates['confidence_score'] = min(
                state.get('confidence_score', 0.5) * 1.2,
                1.0
            )
        else:
            state_updates['confidence_score'] = max(
                state.get('confidence_score', 0.5) * 0.8,
                0.1
            )
        
        # Determine next workflow step
        if validation_results['is_valid'] and validation_results['quality_score'] > 0.7:
            next_step = WorkflowSteps.RESULT_VALIDATION
        else:
            next_step = WorkflowSteps.ERROR_RECOVERY
            state_updates['last_error'] = f"Tool execution quality issues: {validation_results['issues']}"
            state_updates['error_count'] = state.get('error_count', 0) + 1
        
        # Update state through state manager
        await state_manager.transition_workflow_step(
            conversation_id,
            next_step,
            state_updates
        )
        
        logger.info(
            f"Tool execution completed: {execution_results['metadata']['successful_tools']} "
            f"successful, {execution_results['metadata']['failed_tools']} failed → {next_step.value}"
        )
        
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
    """
    Result validation and reflection workflow node.
    
    This node implements comprehensive result validation using the
    MathematicalReasoner component for quality assessment and reflection.
    
    Responsibilities:
    - Mathematical result validation using validation chains
    - Confidence score calculation based on validation results
    - Solution completeness assessment
    - State updates for validation outcomes
    
    Args:
        state: Current mathematical agent state
        reasoner: Mathematical reasoning engine for validation
        state_manager: State manager for state transitions
        
    Returns:
        Dict[str, Any]: State updates containing validation results
        
    Raises:
        ValidationError: If validation process fails
        AgentError: If validation logic encounters errors
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        tool_results = state.get('tool_results', {})
        reasoning_chain = state.get('reasoning_chain', [])
        
        logger.info(f"Starting result validation for {len(tool_results)} tool results")
        
        # Extract expected outcome from reasoning chain
        expected_outcome = {}
        if reasoning_chain:
            latest_reasoning = reasoning_chain[-1]
            expected_outcome = latest_reasoning.get('expected_outcome', {})
        
        # Perform result validation using reasoning engine
        validation_results = await reasoner.validate_results(
            problem=problem,
            reasoning_steps=reasoning_chain,
            tool_results=tool_results,
            expected_outcome=expected_outcome
        )
        
        # Prepare state updates based on validation results
        is_valid = validation_results.get('is_valid', False)
        confidence = validation_results.get('confidence', 0.5)
        completeness = validation_results.get('completeness', 0.5)
        
        state_updates = {
            'confidence_score': confidence,
            'execution_metadata': {
                **state['execution_metadata'],
                'validation_completed': datetime.now().isoformat(),
                'validation_confidence': confidence,
                'solution_completeness': completeness,
                'validation_issues': validation_results.get('identified_errors', [])
            }
        }
        
        # Determine next workflow step based on validation outcome
        if is_valid and confidence > 0.7 and completeness > 0.8:
            # Solution is valid and complete
            next_step = WorkflowSteps.FINAL_RESPONSE
            
            # Prepare final answer from tool results
            final_answer = {}
            for tool_name, result in tool_results.items():
                if isinstance(result, dict) and 'result' in result:
                    final_answer[tool_name] = result['result']
                else:
                    final_answer[tool_name] = result
            
            state_updates['final_answer'] = final_answer
            
        elif confidence < 0.3 or state.get('error_count', 0) > 3:
            # Low confidence or too many errors, try error recovery
            next_step = WorkflowSteps.ERROR_RECOVERY
            state_updates['last_error'] = f"Validation failed: {validation_results.get('identified_errors', [])}"
            state_updates['error_count'] = state.get('error_count', 0) + 1
            
        else:
            # Need more reasoning iterations
            next_step = WorkflowSteps.REASONING
            
        # Update state through state manager
        await state_manager.transition_workflow_step(
            conversation_id,
            next_step,
            state_updates
        )
        
        logger.info(
            f"Validation completed: {'PASSED' if is_valid else 'FAILED'} "
            f"(confidence: {confidence:.2f}, completeness: {completeness:.2f}) → {next_step.value}"
        )
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Result validation failed: {e}", exc_info=True)
        raise AgentError(f"Validation node failed: {str(e)}") from e


@log_function_call(logger)
async def final_response_node(
    state: MathAgentState,
    state_manager: StateManager
) -> Dict[str, Any]:
    """
    Final response generation workflow node.
    
    This node implements final response generation and formatting,
    preparing the complete solution for user presentation.
    
    Responsibilities:
    - Final answer synthesis from tool results and reasoning
    - Response formatting for user presentation
    - Solution summary and explanation generation
    - Workflow completion handling
    
    Args:
        state: Current mathematical agent state
        state_manager: State manager for final state updates
        
    Returns:
        Dict[str, Any]: State updates containing final response
        
    Raises:
        AgentError: If response generation fails
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        final_answer = state.get('final_answer', {})
        tool_results = state.get('tool_results', {})
        confidence_score = state.get('confidence_score', 0.5)
        
        logger.info(f"Generating final response (confidence: {confidence_score:.2f})")
        
        # Generate comprehensive response
        response_components = {
            'problem_statement': problem,
            'solution_approach': state['mathematical_context'].get('strategy', 'analytical'),
            'final_answer': final_answer if final_answer else tool_results,
            'confidence_score': confidence_score,
            'tools_used': list(tool_results.keys()),
            'execution_summary': {
                'total_iterations': state.get('iteration_count', 0),
                'reasoning_steps': len(state.get('reasoning_chain', [])),
                'tools_executed': len(tool_results),
                'execution_time': state['execution_metadata'].get('total_execution_time', 0)
            }
        }
        
        # Format final response message
        if confidence_score > 0.8:
            response_status = "HIGH_CONFIDENCE"
            response_message = "Mathematical problem solved successfully with high confidence."
        elif confidence_score > 0.5:
            response_status = "MEDIUM_CONFIDENCE"
            response_message = "Mathematical problem solved with reasonable confidence."
        else:
            response_status = "LOW_CONFIDENCE"
            response_message = "Mathematical problem processed but results should be verified."
        
        # Prepare final state updates
        state_updates = {
            'workflow_status': 'COMPLETED',
            'final_answer': response_components,
            'execution_metadata': {
                **state['execution_metadata'],
                'workflow_completed': datetime.now().isoformat(),
                'final_confidence': confidence_score,
                'response_status': response_status,
                'total_execution_time': (
                    datetime.now() - 
                    datetime.fromisoformat(state['execution_metadata']['start_time'])
                ).total_seconds()
            }
        }
        
        # Final state update (workflow completion)
        await state_manager.update_state(
            conversation_id,
            state_updates,
            "workflow_completion"
        )
        
        logger.info(
            f"Final response generated: {response_status} "
            f"({len(str(response_components))} chars)"
        )
        
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
    """
    Error recovery workflow node.
    
    This node implements intelligent error recovery using the
    MathematicalReasoner component for alternative strategy generation.
    
    Responsibilities:
    - Error analysis and classification
    - Alternative approach generation using reasoning engine
    - Recovery strategy implementation
    - State updates for recovery attempts
    
    Args:
        state: Current mathematical agent state
        reasoner: Mathematical reasoning engine for recovery strategies
        state_manager: State manager for state transitions
        
    Returns:
        Dict[str, Any]: State updates containing recovery strategy
        
    Raises:
        AgentError: If error recovery fails
    """
    try:
        conversation_id = state['conversation_id']
        problem = state['current_problem']
        error_count = state.get('error_count', 0)
        last_error = state.get('last_error', 'Unknown error')
        reasoning_chain = state.get('reasoning_chain', [])
        
        logger.info(f"Starting error recovery (attempt {error_count + 1}): {last_error}")
        
        # Check if maximum recovery attempts exceeded
        if error_count >= 3:
            logger.warning("Maximum error recovery attempts exceeded, generating fallback response")
            
            # Generate fallback response
            fallback_response = {
                'problem_statement': problem,
                'status': 'PARTIAL_SOLUTION',
                'message': 'Unable to fully solve the problem after multiple attempts.',
                'partial_results': state.get('tool_results', {}),
                'error_summary': last_error,
                'attempts_made': error_count
            }
            
            state_updates = {
                'workflow_status': 'COMPLETED_WITH_ERRORS',
                'final_answer': fallback_response,
                'confidence_score': 0.2,
                'execution_metadata': {
                    **state['execution_metadata'],
                    'recovery_completed': datetime.now().isoformat(),
                    'recovery_status': 'FALLBACK_RESPONSE'
                }
            }
            
            await state_manager.update_state(
                conversation_id,
                state_updates,
                "fallback_response_generation"
            )
            
            return state_updates
        
        # Prepare failed attempts for recovery analysis
        failed_attempts = []
        for reasoning_step in reasoning_chain:
            if reasoning_step.get('confidence', 1.0) < 0.3:
                failed_attempts.append(reasoning_step)
        
        # Generate error recovery strategy using reasoning engine
        recovery_strategy = await reasoner.generate_error_recovery(
            problem=problem,
            failed_attempts=failed_attempts,
            error_analysis={
                'error_message': last_error,
                'error_count': error_count,
                'failed_tools': [
                    tool for tool, result in state.get('tool_results', {}).items()
                    if isinstance(result, dict) and result.get('error')
                ]
            }
        )
        
        # Apply recovery strategy
        alternative_approach = recovery_strategy.get('alternative_approach', 'retry_with_different_method')
        
        state_updates = {
            'mathematical_context': {
                **state['mathematical_context'],
                'strategy': recovery_strategy.get('modified_strategy', {}).get('approach', 'alternative'),
                'recovery_attempt': error_count + 1
            },
            'selected_tools': recovery_strategy.get('tool_alternatives', ['analysis_tool']),
            'confidence_score': recovery_strategy.get('confidence', 0.4),
            'execution_metadata': {
                **state['execution_metadata'],
                'recovery_attempted': datetime.now().isoformat(),
                'recovery_strategy': alternative_approach
            }
        }
        
        # Determine next step based on recovery strategy
        if alternative_approach == 'retry_with_different_method':
            next_step = WorkflowSteps.REASONING
        elif alternative_approach == 'use_different_tools':
            next_step = WorkflowSteps.TOOL_SELECTION
        else:
            next_step = WorkflowSteps.FINAL_RESPONSE
        
        # Update state through state manager
        await state_manager.transition_workflow_step(
            conversation_id,
            next_step,
            state_updates
        )
        
        logger.info(
            f"Error recovery strategy applied: {alternative_approach} → {next_step.value}"
        )
        
        return state_updates
        
    except Exception as e:
        logger.error(f"Error recovery failed: {e}", exc_info=True)
        raise AgentError(f"Error recovery node failed: {str(e)}") from e
